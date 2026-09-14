package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeJournalRecords(t *testing.T, path string, records ...string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(records, "")), 0o600); err != nil {
		t.Fatal(err)
	}
}

func journalRecord(id int64, name string) string {
	return fmt.Sprintf("{\"id\":%d,\"name\":%q,\"data\":\"{}\"}\n", id, name)
}

// writeCommittedJob lays down a job.json whose JournalEventCount commits the
// supplied journal body, which is the shape recovery sees after a clean write.
func writeCommittedJob(t *testing.T, id string, committed int, journal string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), id)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	snapshot := persistedJob{
		ID: id, Status: jobSucceeded, CreatedAt: now, UpdatedAt: now, FinishedAt: now,
		Request:        jobRequest{Filename: "a.wav", AudioFile: "audio.bin"},
		JournalVersion: 1, JournalEventCount: committed,
	}
	if err := writeJSONAtomic(filepath.Join(dir, "job.json"), snapshot, 0o600); err != nil {
		t.Fatal(err)
	}
	if journal != "" {
		writeJournalRecords(t, filepath.Join(dir, "events.ndjson"), journal)
	}
	return dir
}

func TestLoadJobReadsHealthyJournalWithoutWriteAccess(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses the file mode this regression depends on")
	}
	journal := journalRecord(1, "progress") + journalRecord(2, "result")
	dir := writeCommittedJob(t, "readonly", 2, journal)
	path := filepath.Join(dir, "events.ndjson")
	if err := os.Chmod(path, 0o400); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(path, 0o600) })

	loaded, err := loadJob(dir)
	if err != nil {
		t.Fatalf("healthy read-only journal was not loadable: %v", err)
	}
	if len(loaded.events) != 2 || loaded.events[0].Name != "progress" || loaded.events[1].ID != 2 {
		t.Fatalf("events = %#v", loaded.events)
	}
}

func TestLoadEventsRepairsMissingFinalDelimiter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.ndjson")
	body := journalRecord(1, "progress") + strings.TrimSuffix(journalRecord(2, "result"), "\n")
	writeJournalRecords(t, path, body)

	events, err := loadEvents(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || events[1].Name != "result" {
		t.Fatalf("events = %#v", events)
	}
	repaired, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(repaired, []byte(body+"\n")) {
		t.Fatalf("delimiter repair was not persisted: %q", repaired)
	}
}

func TestLoadEventsPersistsTornFinalRecordTruncation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.ndjson")
	complete := journalRecord(1, "progress")
	writeJournalRecords(t, path, complete, `{"id":2,"name":"result","data":`)

	events, err := loadEvents(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %#v", events)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(body, []byte(complete)) {
		t.Fatalf("torn record was not truncated on disk: %q", body)
	}
}

func TestLoadJobPersistsUncommittedTailTruncation(t *testing.T) {
	committed := journalRecord(1, "progress")
	journal := committed + journalRecord(2, "result") + journalRecord(3, "extra")
	dir := writeCommittedJob(t, "uncommitted-tail", 1, journal)

	loaded, err := loadJob(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.events) != 1 || loaded.events[0].Name != "progress" {
		t.Fatalf("uncommitted tail was loaded: %#v", loaded.events)
	}
	body, err := os.ReadFile(filepath.Join(dir, "events.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(body, []byte(committed)) {
		t.Fatalf("uncommitted tail was not truncated on disk: %q", body)
	}
}

// A repair that cannot reach disk must fail the load: the next append seeks to
// the file size and would otherwise write past a record recovery discarded.
func TestLoadEventsFailsWhenRepairCannotBeWritten(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses the file mode this test depends on")
	}
	path := filepath.Join(t.TempDir(), "events.ndjson")
	writeJournalRecords(t, path, journalRecord(1, "progress"), `{"id":2,"name":"result","data":`)
	if err := os.Chmod(path, 0o400); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(path, 0o600) })

	if _, err := loadEvents(path); err == nil {
		t.Fatal("torn journal loaded despite an unapplicable repair")
	}
}

func TestLoadEventsCommittedPreservedErrorCases(t *testing.T) {
	cases := []struct {
		name      string
		journal   string
		absent    bool
		committed int
	}{
		{name: "missing journal with committed events", absent: true, committed: 1},
		{
			name:      "fewer events than the snapshot commits",
			journal:   journalRecord(1, "progress"),
			committed: 2,
		},
		{
			name:      "corruption before the final record",
			journal:   journalRecord(1, "progress") + "{not-json}\n" + journalRecord(2, "result"),
			committed: -1,
		},
		{
			name:      "non-contiguous event ids",
			journal:   journalRecord(1, "progress") + journalRecord(3, "result"),
			committed: -1,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "events.ndjson")
			if !testCase.absent {
				writeJournalRecords(t, path, testCase.journal)
			}
			if _, err := loadEventsCommitted(path, testCase.committed); err == nil {
				t.Fatal("load succeeded, want error")
			}
		})
	}
}

func TestLoadEventsLegacyModeKeepsWholeJournal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.ndjson")
	body := journalRecord(1, "progress") + journalRecord(2, "result") + journalRecord(3, "extra")
	writeJournalRecords(t, path, body)

	events, err := loadEvents(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 3 {
		t.Fatalf("legacy load dropped events: %#v", events)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, []byte(body)) {
		t.Fatalf("legacy load mutated the journal: %q", after)
	}
}

func TestLoadEventsCommittedIgnoresMissingJournalWithoutCommittedEvents(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.ndjson")
	events, err := loadEventsCommitted(path, 0)
	if err != nil || len(events) != 0 {
		t.Fatalf("loadEventsCommitted = %#v, %v; want no events and no error", events, err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("absent journal was created: %v", err)
	}
}

// Appends create the journal on the first event and extend it thereafter; only
// the create syncs the directory, so the extend path needs explicit coverage.
func TestAppendEventsCreatesThenExtendsJournal(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "appends")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	jb := &job{
		id: "appends", dir: dir, createdAt: now, updatedAt: now, status: jobRunning,
		request: jobRequest{AudioFile: "audio.bin"},
	}
	if err := os.WriteFile(filepath.Join(dir, "audio.bin"), []byte("audio"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := jb.persistLocked(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "events.ndjson")
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("journal existed before the first append: %v", err)
	}

	names := []string{"first", "second", "third"}
	for index, name := range names {
		if err := jb.push(name, map[string]any{"index": index}); err != nil {
			t.Fatalf("push %s: %v", name, err)
		}
	}

	loaded, err := loadJob(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.events) != len(names) {
		t.Fatalf("events = %#v", loaded.events)
	}
	for index, name := range names {
		event := loaded.events[index]
		if event.ID != int64(index+1) || event.Name != name {
			t.Fatalf("event %d = %#v", index, event)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasSuffix(body, []byte("\n")) || bytes.Count(body, []byte("\n")) != len(names) {
		t.Fatalf("journal is not a well-formed ndjson stream: %q", body)
	}
}
