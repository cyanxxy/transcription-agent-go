package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/cyanxxy/transcription-agent-go/internal/agents"
	"github.com/cyanxxy/transcription-agent-go/internal/models"
	"github.com/cyanxxy/transcription-agent-go/internal/obs"
	"github.com/cyanxxy/transcription-agent-go/internal/workflow"
)

func (s *server) startJobWorkers(recoverable []*job) {
	capacity := s.maxConcurrency + s.maxQueued
	if capacity < 1 {
		capacity = 1
	}
	if len(recoverable) > capacity {
		capacity = len(recoverable)
	}
	s.queue = make(chan *job, capacity)
	s.workerCtx, s.stopWorkers = context.WithCancel(context.Background())
	s.admissionMu.Lock()
	s.admitted = len(recoverable)
	s.admissionMu.Unlock()
	for worker := 0; worker < s.maxConcurrency; worker++ {
		s.jobWG.Add(1)
		go func(workerID int) {
			defer s.jobWG.Done()
			s.jobWorker(workerID)
		}(worker)
	}
	for _, jb := range recoverable {
		s.queue <- jb
	}
}

func (s *server) jobWorker(workerID int) {
	for {
		select {
		case <-s.workerCtx.Done():
			return
		case jb := <-s.queue:
			if jb == nil {
				continue
			}
			if s.workerCtx.Err() != nil {
				return
			}
			s.executeJob(workerID, jb)
		}
	}
}

func (s *server) executeJob(workerID int, jb *job) {
	defer s.releaseAdmission()
	jobCtx, cancel := context.WithCancel(s.workerCtx)
	defer cancel()
	jobCtx = obs.WithLogger(jobCtx, s.logger.With(
		slog.String("request_id", jb.request.RequestID),
		slog.String("job_id", jb.id),
		slog.Int("worker_id", workerID),
	))
	attempt, status, startErr := jb.startExecution(cancel)
	if status == jobCancelRequested {
		_, _ = finalizeCanceledJob(jb, false, "job canceled before execution")
		return
	}
	if status != jobRunning {
		return
	}
	if startErr != nil {
		_, _ = jb.transitionWithEventsFrom(
			jobRunning,
			jobFailed,
			"could not persist job start",
			time.Now().UTC(),
			jobEvent{name: "error-event", data: map[string]any{"message": "could not persist job start", "request_id": jb.request.RequestID}},
		)
		return
	}
	s.registerJob(jb.id, cancel)
	defer func() {
		s.unregisterJob(jb.id)
		jb.mu.Lock()
		jb.cancel = nil
		jb.mu.Unlock()
	}()
	defer func() {
		if recovered := recover(); recovered != nil {
			obs.LoggerFrom(jobCtx).Error("job worker panicked", "panic", recovered, "stack", string(debug.Stack()))
			if canceled, _ := finalizeCanceledJob(jb, false, "job canceled"); canceled {
				return
			}
			_, _ = jb.transitionWithEventsFrom(
				jobRunning,
				jobFailed,
				"worker panic",
				time.Now().UTC(),
				jobEvent{name: "error-event", data: map[string]any{"message": "internal error", "request_id": jb.request.RequestID}},
			)
		}
	}()
	obs.LoggerFrom(jobCtx).Debug("job started", "attempt", attempt)
	runCtx, timeoutCancel := context.WithTimeout(jobCtx, 30*time.Minute)
	defer timeoutCancel()
	progress := func(stage string, fraction float64) {
		if err := jb.push("progress", map[string]any{"stage": stage, "fraction": fraction}); err != nil {
			obs.LoggerFrom(jobCtx).Error("persist progress event", "error", err)
			cancel()
		}
	}
	result, err := s.transcribePersistedJob(runCtx, jb, progress)
	if err != nil {
		if errors.Is(err, context.Canceled) && !s.ready.Load() {
			if canceled, cancelErr := finalizeCanceledJob(jb, false, "job canceled during shutdown"); canceled || cancelErr != nil {
				return
			}
			requeued, requeueErr := jb.transitionFrom(jobRunning, jobQueued, "")
			if requeueErr != nil {
				obs.LoggerFrom(jobCtx).Error("persist interrupted job for restart", "error", requeueErr)
			}
			if !requeued {
				_, _ = finalizeCanceledJob(jb, false, "job canceled during shutdown")
			}
			return
		}
		if canceled, cancelErr := finalizeCanceledJob(jb, errors.Is(err, context.Canceled), "job canceled"); canceled || cancelErr != nil {
			return
		}
		obs.LoggerFrom(jobCtx).Error("transcription failed", "error", err.Error())
		_, _ = jb.transitionWithEventsFrom(
			jobRunning,
			jobFailed,
			"",
			time.Now().UTC(),
			jobEvent{name: "error-event", data: map[string]any{"message": publicErrorMessage(err), "request_id": jb.request.RequestID}},
		)
		return
	}
	srt, _ := workflow.ExportTranscript(result, "srt")
	txt, _ := workflow.ExportTranscript(result, "txt")
	at := time.Now().UTC()
	target := jobSucceeded
	events := []jobEvent{{name: "result", data: map[string]any{
		"result": result, "formatted_text": txt, "srt": srt, "job_id": jb.id,
	}}}
	if result.AgentRun != nil && result.AgentRun.HumanReview.Status == "required" {
		target = jobAwaitingReview
		events = append(events, jobEvent{name: "review-required", data: map[string]any{
			"job_id": jb.id, "reasons": result.AgentRun.HumanReview.Reasons,
			"review_expires_at": at.Add(s.jobTTL).Format(time.RFC3339Nano),
		}})
	}
	completed, completeErr := jb.transitionWithEventsFrom(jobRunning, target, "", at, events...)
	if completeErr != nil {
		obs.LoggerFrom(jobCtx).Error("persist job completion", "error", completeErr)
		return
	}
	if !completed {
		_, _ = finalizeCanceledJob(jb, false, "job canceled")
	}
}

func finalizeCanceledJob(jb *job, allowRunning bool, message string) (bool, error) {
	completed, err := jb.transitionWithEventsFrom(
		jobCancelRequested,
		jobCanceled,
		"",
		time.Now().UTC(),
		jobEvent{name: "canceled", data: map[string]any{"message": message}},
	)
	if completed || err != nil || !allowRunning {
		return completed, err
	}
	return jb.transitionWithEventsFrom(
		jobRunning,
		jobCanceled,
		"",
		time.Now().UTC(),
		jobEvent{name: "canceled", data: map[string]any{"message": message}},
	)
}

func (s *server) transcribePersistedJob(ctx context.Context, jb *job, progress workflow.ProgressFn) (*models.TranscriptResult, error) {
	audioPath, err := persistedAudioPath(jb)
	if err != nil {
		return nil, err
	}
	wfl, err := workflow.New(s.apiKey, s.buildOptionsFromValues(jb.request.Form)...)
	if err != nil {
		return nil, err
	}
	defer wfl.Deps.Cleanup()
	if strings.TrimSpace(s.parakeet) != "" {
		if sidecar, sidecarErr := agents.ParakeetFromDeps(wfl.Deps.Transcription, s.parakeet); sidecarErr == nil {
			wfl.WithParakeet(sidecar)
		}
	}
	if s.skillsReg != nil {
		wfl.WithSkills(s.skillsReg)
	}
	return wfl.Transcribe(ctx, workflow.TranscribeInput{
		FilePath:     audioPath,
		Filename:     jb.request.Filename,
		CustomPrompt: jb.request.Form["custom_prompt"],
		UserContext:  buildUserContextFromValues(jb.request.Form),
		Progress:     progress,
		RunFinished: func(run *models.AgentRun) {
			if err := jb.push("agent-run", map[string]any{"run": run}); err != nil {
				obs.LoggerFrom(ctx).Error("persist agent run", "error", err)
			}
		},
	})
}

func persistedAudioPath(jb *job) (string, error) {
	if jb == nil || strings.TrimSpace(jb.dir) == "" {
		return "", errors.New("persisted job directory is missing")
	}
	if jb.request.AudioFile != "audio.bin" {
		return "", fmt.Errorf("persisted audio file must be audio.bin, got %q", jb.request.AudioFile)
	}
	path := filepath.Join(jb.dir, "audio.bin")
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("inspect persisted audio: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return "", errors.New("persisted audio must be a regular file")
	}
	return path, nil
}
