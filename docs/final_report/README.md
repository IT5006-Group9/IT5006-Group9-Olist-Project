# Final report (Phase 3)

`Team9_Phase3_IT5006_AY2627Sem1.docx` is the final report: cover, table of contents, main text (12 pt), references, GitHub section, AI declaration and appendices A–C. Its structure follows the course's Milestone 3 sample reports (Executive Summary; Introduction & Background; Methodology; Results & Analysis incl. ethical considerations; Conclusion with limitations and future work), plus the brief's required deployment, business-impact and feedback-correction sections.

| File | Role |
|---|---|
| `report.md` | main text (Markdown subset) |
| `cover.json` | cover page: title, members, links, date |
| `assemble.py` | writes `appendix.md` from the repository's saved outputs and `docs/report_appendix_features_models.md` |
| `build.js` | converts `report.md` + `appendix.md` to `.docx` (docx-js) |
| `topdf.ps1` | optional: updates the table of contents in Word and exports a PDF |
| `eda.py` | draws `figures/eda_patterns.png` from the raw CSVs |
| `figures/` | figures used in the report (copied from `outputs/`) |

Rebuild (from the repository root; needs Node.js and `npm install docx@9` in this folder):

```bash
python docs/final_report/assemble.py
node docs/final_report/build.js docs/final_report/report.md docs/final_report/appendix.md docs/final_report/Team9_Phase3_IT5006_AY2627Sem1.docx
powershell -File docs/final_report/topdf.ps1 <full path to .docx> <full path to .pdf>
```

Edit the text in `report.md` (or directly in Word after the last rebuild). Every number in the report comes from the saved tables in `outputs/` and `docs/`.
