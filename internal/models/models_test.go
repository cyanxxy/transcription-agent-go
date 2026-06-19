package models

import "testing"

func TestParseTimestampSeconds(t *testing.T) {
	tests := []struct {
		in   string
		want float64
		ok   bool
	}{
		{"[00:00:00]", 0, true},
		{"[00:01:30]", 90, true},
		{"[01:00:00]", 3600, true},
		{"[12:34:56]", 12*3600 + 34*60 + 56, true},
		{"00:01:30", 0, false},
		{"[xx:yy:zz]", 0, false},
		{"[1:1:1]", 0, false},
	}
	for _, tt := range tests {
		got, err := ParseTimestampSeconds(tt.in)
		if tt.ok && err != nil {
			t.Errorf("ParseTimestampSeconds(%q) unexpected error: %v", tt.in, err)
			continue
		}
		if !tt.ok && err == nil {
			t.Errorf("ParseTimestampSeconds(%q) expected error, got %v", tt.in, got)
			continue
		}
		if tt.ok && got != tt.want {
			t.Errorf("ParseTimestampSeconds(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestFormatTimestamp(t *testing.T) {
	cases := map[float64]string{
		0:    "[00:00:00]",
		59:   "[00:00:59]",
		60:   "[00:01:00]",
		3601: "[01:00:01]",
		3661: "[01:01:01]",
		-5:   "[00:00:00]",
	}
	for in, want := range cases {
		if got := FormatTimestamp(in); got != want {
			t.Errorf("FormatTimestamp(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestAdjustTimestamp(t *testing.T) {
	if got := AdjustTimestamp("[00:00:30]", 90); got != "[00:02:00]" {
		t.Errorf("AdjustTimestamp = %s, want [00:02:00]", got)
	}
	if got := AdjustTimestamp("[invalid]", 5); got != "[invalid]" {
		t.Errorf("AdjustTimestamp should keep invalid input intact, got %s", got)
	}
}

func TestTranscriptSegmentValidate(t *testing.T) {
	cases := []struct {
		name string
		seg  TranscriptSegment
		ok   bool
	}{
		{"valid", TranscriptSegment{Timestamp: "[00:00:00]", Speaker: "Alice", Text: "Hi"}, true},
		{"bad timestamp", TranscriptSegment{Timestamp: "00:00:00", Speaker: "Alice", Text: "Hi"}, false},
		{"minutes out of range", TranscriptSegment{Timestamp: "[00:99:00]", Speaker: "Alice", Text: "Hi"}, false},
		{"empty speaker", TranscriptSegment{Timestamp: "[00:00:00]", Speaker: " ", Text: "Hi"}, false},
		{"empty text", TranscriptSegment{Timestamp: "[00:00:00]", Speaker: "Alice", Text: " "}, false},
	}
	for _, tc := range cases {
		err := tc.seg.Validate()
		if tc.ok && err != nil {
			t.Errorf("%s: expected ok, got %v", tc.name, err)
		}
		if !tc.ok && err == nil {
			t.Errorf("%s: expected error, got nil", tc.name)
		}
	}
}

func TestUniqueSpeakers(t *testing.T) {
	r := &TranscriptResult{
		Segments: []TranscriptSegment{
			{Speaker: "Alice"},
			{Speaker: "Bob"},
			{Speaker: "Alice"},
		},
	}
	out := r.UniqueSpeakers()
	if len(out) != 2 {
		t.Fatalf("expected 2 unique speakers, got %v", out)
	}
}

func TestNormalizeTimestamp(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"single digit hour", "[1:05:30]", "[01:05:30]"},
		{"already canonical", "[01:05:30]", "[01:05:30]"},
		{"all single digit", "[1:2:3]", "[01:02:03]"},
		{"long hours unchanged", "[100:00:00]", "[100:00:00]"},
		{"garbage unchanged", "garbage", "garbage"},
		{"two field form unchanged", "[00:00]", "[00:00]"},
	}
	for _, tt := range tests {
		if got := NormalizeTimestamp(tt.in); got != tt.want {
			t.Errorf("%s: NormalizeTimestamp(%q) = %q, want %q", tt.name, tt.in, got, tt.want)
		}
	}
}

func TestValidateAcceptsNormalizedSingleDigitHour(t *testing.T) {
	normalized := TranscriptSegment{
		Timestamp: NormalizeTimestamp("[1:05:30]"),
		Speaker:   "A",
		Text:      "hi",
	}
	if err := normalized.Validate(); err != nil {
		t.Errorf("Validate() on normalized timestamp %q: unexpected error: %v", normalized.Timestamp, err)
	}

	// The raw, un-normalized timestamp must fail Validate, documenting why
	// NormalizeTimestamp is required before validation.
	raw := TranscriptSegment{
		Timestamp: "[1:05:30]",
		Speaker:   "A",
		Text:      "hi",
	}
	if err := raw.Validate(); err == nil {
		t.Errorf("Validate() on raw timestamp %q: expected error, got nil", raw.Timestamp)
	}
}

func TestFormatTimestampLongHoursRoundTrip(t *testing.T) {
	const longSeconds = 360000.0
	formatted := FormatTimestamp(longSeconds)
	if formatted != "[100:00:00]" {
		t.Fatalf("FormatTimestamp(%v) = %q, want [100:00:00]", longSeconds, formatted)
	}
	if !timestampRE.MatchString(formatted) {
		t.Errorf("timestampRE should match %q", formatted)
	}
	secs, err := ParseTimestampSeconds(formatted)
	if err != nil {
		t.Fatalf("ParseTimestampSeconds(%q) unexpected error: %v", formatted, err)
	}
	if secs != longSeconds {
		t.Errorf("ParseTimestampSeconds(%q) = %v, want %v", formatted, secs, longSeconds)
	}

	// Adjusting past 99h must produce something that round-trips back through
	// ParseTimestampSeconds.
	adjusted := AdjustTimestamp("[99:00:00]", 7200)
	if !timestampRE.MatchString(adjusted) {
		t.Errorf("AdjustTimestamp past 99h produced %q which timestampRE rejects", adjusted)
	}
	if _, err := ParseTimestampSeconds(adjusted); err != nil {
		t.Errorf("ParseTimestampSeconds(%q) after AdjustTimestamp past 99h: unexpected error: %v", adjusted, err)
	}
}

func TestUniqueSpeakersFirstAppearanceOrder(t *testing.T) {
	r := &TranscriptResult{
		Segments: []TranscriptSegment{
			{Speaker: "Zoe"},
			{Speaker: "Alice"},
			{Speaker: "Zoe"},
			{Speaker: "Bob"},
		},
	}
	got := r.UniqueSpeakers()
	want := []string{"Zoe", "Alice", "Bob"}
	if len(got) != len(want) {
		t.Fatalf("UniqueSpeakers() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("UniqueSpeakers() = %v, want %v (first-appearance order, not alphabetical)", got, want)
		}
	}
}

func TestFormattedText(t *testing.T) {
	r := &TranscriptResult{
		Segments: []TranscriptSegment{
			{Timestamp: "[00:00:00]", Speaker: "Alice", Text: "Hello"},
			{Timestamp: "[00:00:05]", Speaker: "Bob", Text: "Hi"},
		},
	}
	got := r.FormattedText()
	want := "[00:00:00] Alice: Hello\n[00:00:05] Bob: Hi"
	if got != want {
		t.Errorf("FormattedText() =\n%q\nwant\n%q", got, want)
	}
}
