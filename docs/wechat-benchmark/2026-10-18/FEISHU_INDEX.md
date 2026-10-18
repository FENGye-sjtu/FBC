> 2026-10-18云端版本核对更新：两套目录全部文件路径与清风原ZIP对应，无新增或缺失；68份Markdown源文本（每套34份）全文一致，仅统一换行和末尾空行。另8个非Markdown文件当前云端内容未核验，不能声称全部逐字节一致。详见CLOUD_VERSION_VERIFICATION.md、CLOUD_VERSION_VERIFICATION.json及LATEST_CLOUD_TEXT.txt；本段更新取代下面旧版的云端核验进度说明。

[云端核对报告](CLOUD_VERSION_VERIFICATION.md) · [当前Markdown全文](LATEST_CLOUD_TEXT.txt)

# 飞书与学生资料索引

[下载包含微信与学生资料的整合包](https://github.com/FENGye-sjtu/FBC/releases/download/wechat-benchmark-2026-10-18/wechat-benchmark-2026-10-18-with-student-materials.zip) · [只下载学生及任务补充包](https://github.com/FENGye-sjtu/FBC/releases/download/wechat-benchmark-2026-10-18/feishu-student-materials-2026-10-18.zip) · [GitHub Release](https://github.com/FENGye-sjtu/FBC/releases/tag/wechat-benchmark-2026-10-18)

所有路径均为整合ZIP内相对路径。只需GitHub，无需再访问飞书。

|资料|ZIP内位置或GitHub文本|
|---|---|
|24条聊天消息及嵌套关系|CHAT_TRANSCRIPT.txt、messages.json|
|两套企业需求原件|attachments/欢乐互娱-企业需求-学生发布版.zip（无OKR）；attachments/欢乐互娱-企业需求.zip（带OKR）|
|两套任务资料展开版|feishu/task-materials/；[全文](TASK_MATERIALS.txt)|
|学生表与5条项目提交|feishu/学生总提交记录.xlsx、feishu/欢乐互娱提交记录.json|
|4份说明全文及方术伟包内说明|[STUDENT_DOCS.txt](STUDENT_DOCS.txt)|
|新增14个原文件校验与映射|[STUDENT_FILE_MANIFEST.json](STUDENT_FILE_MANIFEST.json)|
|递归ZIP内部路径和SHA256|feishu/inventories/；路径中的 !/ 表示再解压一层|

## 学生原文件

### 杨柏森

- `feishu/student-submissions/杨柏森/FDE-X-T01-年度战略规划-Skill包-杨柏森.zip` — 587148字节，SHA256 `0fa655ee3d0e5ab05666d0a922db6b5f5677338a1d7b82c96e8d7363db1cacb0`
- `feishu/student-submissions/杨柏森/项目说明与验证记录-杨柏森.docx` — 19422字节，SHA256 `5394118900586105566494aaae37b58e1dc08be4c5a17c7897852beec9172974`

### 赵一然

- `feishu/student-submissions/赵一然/00-FDE-X-Skill交付总包.zip` — 59514962字节，SHA256 `3a808fb1d9e68010d7b4b6fa49b9bb3bc62f76240b8e941e04ea89462d43b593`
- `feishu/student-submissions/赵一然/T01 项目说明与验证记录（星芽英语探险·完全虚构案例）.docx` — 15995字节，SHA256 `b5efcb1dab5707c38e48b5480aee83cdff73373d4fc2907ac3e6631698dbdf1a`

### 方术伟

- `feishu/student-submissions/方术伟/FDEX.zip` — 3157552字节，SHA256 `92489777164da015f1713748a5e60397cbb05229d41971c6d9721fbe0a91b8a3`

### 陈思微

- `feishu/student-submissions/陈思微/strategy-agent.zip` — 239991字节，SHA256 `fcacd462755d076217d78040c66e8c394b901ac1d82619f9b9b0273ee6b4dadf`
- `feishu/student-submissions/陈思微/战略 Agent｜项目说明与验证记录.docx` — 6096字节，SHA256 `d60118f75e8c2d227794599e68e8c71c4770fc2798e9730a677a16b573aa82ab`

### 王奕鸣

- `feishu/student-submissions/王奕鸣/01-Skills-and-ordinary-components.zip` — 194106字节，SHA256 `183094abba5d962f16fdfe310ae323db6d38796ccdcbc7fbc7c0ae92dc2472af`
- `feishu/student-submissions/王奕鸣/03-Strategy-1.1.0-unpackaged-Skill-development-materials.zip` — 38517字节，SHA256 `8cce3bd866dc9b6e181cef8ccc2aaefe252f33956122a19e55256b9cb4b928e6`
- `feishu/student-submissions/王奕鸣/04-PPT-2.5.13-candidate-unpackaged-Skill-development-materials.zip` — 104960字节，SHA256 `024163ebc2a8819dfa7e6baf013763237ab08df8b6282feb4a789498be74d8fe`
- `feishu/student-submissions/王奕鸣/conditional-annual-strategy-draft.zip` — 40966字节，SHA256 `a2ca339c1613f65b01bf464c9ab1376951ca277e424318146f6f915977059b4e`
- `feishu/student-submissions/王奕鸣/editable-ppt-workflow-draft.zip` — 110180字节，SHA256 `68a83170a1995a14543f803dabce618beea9083514253c15738d284de39c6b5d`
- `feishu/student-submissions/王奕鸣/Clean07-finite-outcomes-and-legacy-PPT-recipient-review.zip` — 851897字节，SHA256 `696213a2935bbedad6aa6df118a5ffe8368ec0dc4217179aa77a187e62dcb12d`
- `feishu/student-submissions/王奕鸣/欢乐互娱年度战略规划：项目说明与验证记录.docx` — 48291字节，SHA256 `b22451cd9badb9788c2d3b7777450393e02c0e13558e3fffeae83c0564d531b6`

## 范围与核验

飞书补充资料（2026-10-18）

按用户确认的范围，已收集5位学生的10个skill ZIP与4个项目说明DOCX；方术伟说明见FDEX.zip。14个原件的字节数、SHA256和递归ZIP校验结果见manual-downloads-manifest.json；递归文件清单见inventories/。DOCX原件保留，另提取全文、外部链接及嵌入图片。

学生总表导出原件和本项目5条记录保留于本目录；右侧独立session列的10个附件经用户明确排除，不再作为缺件，也不声称已经下载。skill包内部自带会话数据保留。

两套飞书任务入口分别对应微信中的“欢乐互娱-企业需求-学生发布版.zip”和“欢乐互娱-企业需求.zip”。前者无OKR，后者带OKR。学生版云端根README和需求提报表与微信ZIP逐字节一致；其余云端文件和评测目录未逐文件比对，不能据此声称云端最新版本完全相同。本包使用完整、已通过CRC校验的微信原ZIP，并在task-materials/展开供读取，文本汇总见包根目录TASK_MATERIALS.txt。

开发资料和报告含模拟案例。保留原文不代表对学生声明、任务验收、PPT质量或实验结论作出独立验证。王奕鸣说明明确保留未完成/失败状态，不应概括成全部通过。
