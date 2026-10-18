# 清风原 ZIP 与飞书当前版本核对

核对日期：2026-10-18（北京时间）。核对的是两个给定飞书目录当时可见的文件，不代表持续同步或历史版本审计。

学生版对应 `欢乐互娱-企业需求-学生发布版.zip`；评测版对应 `欢乐互娱-企业需求.zip`。

- 已遍历两套目录共20个文件夹，学生版41个文件、评测版40个文件；全部路径与各自原ZIP相符，未发现新增或缺失。
- 排除5个 macOS `.DS_Store` 元数据后，共76个有效资料文件。
- **68份 Markdown（每套34份）已逐份通过“查看代码”视图复制全文并比较，全部正文一致。** 比较仅统一CRLF/LF和忽略末尾空行；没有忽略正文、空格或标点差异。全文见 [LATEST_CLOUD_TEXT.txt](LATEST_CLOUD_TEXT.txt)，逐项状态和哈希见 [JSON核对清单](CLOUD_VERSION_VERIFICATION.json)。
- **8个非Markdown文件的当前云端内容尚未核验**：每套的 manifest.json、K03-路线图.svg、U08-访谈台账.csv 和年度战略草案-橙白框架样板.pptx。原文件下载通道未完成，打开页面中明确显示的下载地址时浏览器返回 `net::ERR_BLOCKED_BY_CLIENT`；没有绕过拦截。这8项在清风原ZIP及既有GitHub整合包中都有保留，但不能据此断言与当前飞书原件逐字节一致。

因此，可确认“文件清单完全对应、68份Markdown正文一致”；**尚不能确认“两份ZIP与飞书当前版本全部内容完全一致”**。目录修改日期、名称相同不作为内容一致的证据。

未核验项：

- `student / 学生材料/模拟知识库/manifest.json`（原ZIP SHA256：`5c2871c248962876376783adad04af0275d5626ba84a4b21f150af5c2c29f88c`）
- `student / 学生材料/输出模板/年度战略草案-橙白框架样板.pptx`（原ZIP SHA256：`a670e4b8560206fc5de379491582182b3a5ea3acc34ab845d7b545c0f07d7365`）
- `evaluation / 学生材料/模拟知识库/manifest.json`（原ZIP SHA256：`5c2871c248962876376783adad04af0275d5626ba84a4b21f150af5c2c29f88c`）
- `evaluation / 学生材料/输出模板/年度战略草案-橙白框架样板.pptx`（原ZIP SHA256：`a670e4b8560206fc5de379491582182b3a5ea3acc34ab845d7b545c0f07d7365`）
- `student / 学生材料/模拟知识库/资源/K03-路线图.svg`（原ZIP SHA256：`e606d37b125a0ca56d5d5ecf59eb3abe9ec2c5a95f8682a5e86149431705be8f`）
- `student / 学生材料/模拟知识库/资源/U08-访谈台账.csv`（原ZIP SHA256：`417687e50159dff4ccae679c034d64a047b578ddd3d128168f21c904983f1268`）
- `evaluation / 学生材料/模拟知识库/资源/K03-路线图.svg`（原ZIP SHA256：`e606d37b125a0ca56d5d5ecf59eb3abe9ec2c5a95f8682a5e86149431705be8f`）
- `evaluation / 学生材料/模拟知识库/资源/U08-访谈台账.csv`（原ZIP SHA256：`417687e50159dff4ccae679c034d64a047b578ddd3d128168f21c904983f1268`）

原始微信聊天、两个需求ZIP、学生10个skill ZIP及4份说明DOCX均已归档，入口见 [README](README.md) 和 [FEISHU_INDEX](FEISHU_INDEX.md)。学生表右侧独立session附件依用户要求排除。既有整合包内旧版说明保留为当时快照；本次云端核对进展以本报告为准。
