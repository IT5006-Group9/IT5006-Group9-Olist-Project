param([string]$in, [string]$out)
# Update the table of contents and fields, save the .docx, then export PDF with Word (wdExportFormatPDF = 17).
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$d = $w.Documents.Open($in)
foreach ($t in $d.TablesOfContents) { $t.Update() }
$d.Fields.Update() | Out-Null
$d.Save()
$d.ExportAsFixedFormat($out, 17)
try { $d.Close($false); $w.Quit() } catch { }   # Word sometimes exits by itself after export
