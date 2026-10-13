package export

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"strconv"
	"strings"

	"github.com/cyanxxy/transcription-agent-go/internal/models"
)

// renderDOCX writes a minimal Office Open XML document: a title, a summary
// line, then one paragraph per speaker turn with the speaker in bold and an
// optional muted timestamp.
func renderDOCX(result *models.TranscriptResult, segments []models.TranscriptSegment, opts Options) ([]byte, error) {
	var body strings.Builder
	body.WriteString(paragraph("Title", run(opts.Title, runStyle{})))
	if summary := documentSummary(result, segments); summary != "" {
		body.WriteString(paragraph("Subtitle", run(summary, runStyle{})))
	}
	for _, t := range speakerTurns(segments) {
		heading := run(t.speaker, runStyle{bold: true})
		if opts.IncludeTimestamps {
			heading += run("   "+clock(t.start), runStyle{color: "6B7280", size: 18})
		}
		body.WriteString(paragraph("Speaker", heading))
		body.WriteString(paragraph("", run(strings.Join(t.texts, " "), runStyle{})))
	}

	modified := result.CreatedAt
	if modified.IsZero() {
		modified = epoch
	}
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	files := []struct{ name, content string }{
		{"[Content_Types].xml", docxContentTypes},
		{"_rels/.rels", docxRootRels},
		{"word/_rels/document.xml.rels", docxDocumentRels},
		{"word/styles.xml", docxStyles},
		{"word/document.xml", docxDocumentStart + body.String() + docxDocumentEnd},
	}
	for _, file := range files {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: file.name, Method: zip.Deflate, Modified: modified})
		if err != nil {
			return nil, err
		}
		if _, err := w.Write([]byte(file.content)); err != nil {
			return nil, err
		}
	}
	if err := zw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

type runStyle struct {
	bold  bool
	color string
	// size is in half-points.
	size int
}

func run(text string, style runStyle) string {
	var props strings.Builder
	if style.bold {
		props.WriteString("<w:b/>")
	}
	if style.color != "" {
		props.WriteString(`<w:color w:val="` + style.color + `"/>`)
	}
	if style.size > 0 {
		props.WriteString(`<w:sz w:val="` + strconv.Itoa(style.size) + `"/>`)
	}
	out := "<w:r>"
	if props.Len() > 0 {
		out += "<w:rPr>" + props.String() + "</w:rPr>"
	}
	return out + `<w:t xml:space="preserve">` + escapeXML(text) + "</w:t></w:r>"
}

func paragraph(style, runs string) string {
	if style == "" {
		return "<w:p>" + runs + "</w:p>"
	}
	return `<w:p><w:pPr><w:pStyle w:val="` + style + `"/></w:pPr>` + runs + "</w:p>"
}

func escapeXML(s string) string {
	var buf bytes.Buffer
	// EscapeText also replaces characters that are invalid in XML 1.0.
	_ = xml.EscapeText(&buf, []byte(s))
	return buf.String()
}

const docxContentTypes = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>`

const docxRootRels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`

const docxDocumentRels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>`

const docxStyles = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="40"/><w:szCs w:val="40"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="360"/></w:pPr><w:rPr><w:color w:val="6B7280"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Speaker"><w:name w:val="Speaker"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="40"/></w:pPr></w:style></w:styles>`

const docxDocumentStart = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>`

const docxDocumentEnd = `<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>`
