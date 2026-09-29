# 中文毕业论文答辩材料

19页、16:9。PPTX正文可编辑，逐页备注随文件保存；SVG科学图带PNG兼容回退。PDF从同一layout.json版式构建，已有科学图以原PDF矢量对象嵌入，照片使用完整原始PNG。学校、学位、专业及个人姓名未推测填写。

- `semantic_geometric_mapping_defense.pptx`：可编辑答辩稿，建议安装Noto Sans CJK SC。
- `semantic_geometric_mapping_defense.pdf`：与PPT共用版式数据的展示PDF。
- `preview/slide_*.png`、`contact_sheet.png`：PDFium实际渲染预览。
- `speaker_notes.md` / `qa_outline.md`：逐页讲稿 / 18项问答。
- `layout.json`、`plotted_data.json`、`manifest.json`：版式、数字及输入输出绑定。

本环境无Microsoft Office或LibreOffice。已检查PPT包内19张幻灯片、19份备注、原生文字、SVG和全部关系目标；**没有声称PPT经过Office渲染**。PDF与PNG是同源独立排版的实际渲染。正式答辩前可在学校使用的软件中打开PPT检查字体；需要稳定显示时可直接使用PDF。

新布局8条结果全部纳入：1胜3平，约+2.98%；与原确认+6.16%及完整纠错+2.62%分开。36场景级任务保留原流程35合格/1失败身份和9组S/B持平。照片不作为实车性能证据。

重建到一个全新目录（原文件不覆盖）：
```bash
PYTHONPATH=tmp/final-writing-python:tmp/thesis-python_20260928/site-packages python3 scripts/build_final_defense_20260929.py --output /tmp/nso-defense-rebuild
```

所有输出仅复用保存数据与现有图，不运行World、TSDF、评价或策略。
