# 爱合伙 · 社群匹配与公共 API 查询 Playbook

## 匹配逻辑

当收集到用户的创业画像（角色、方向、城市、阶段与诉求）后，智能体将利用以下 API 对接爱合伙官网数据资源：

1. **寻找合伙人 (Cofounder matching)**:
   - 接口：`GET https://www.aihehuo.com/ai/users.json` 或 `Accept: text/markdown` 请求 `.md` 格式。
   - 参数：`keyword=<city>,<role>` 或 `q=<semantic_phrase>`。
   - 示例：`https://www.aihehuo.com/ai/users.json?keyword=杭州,开发`。

2. **寻找微信社群 (WeChat group matching)**:
   - 接口：`GET https://www.aihehuo.com/ai/wechat_groups.json`。
   - 参数：`keyword=<city_or_industry>`。
   - 示例：`https://www.aihehuo.com/ai/wechat_groups.json?keyword=上海`。

3. **解答规则与常见问题 (FAQ matching)**:
   - 接口：`GET https://www.aihehuo.com/ai/faqs.json`。
   - 参数：`keyword=<keyword>`。
   - 示例：`https://www.aihehuo.com/ai/faqs.json?keyword=加V认证`。

## 安全与隐私底线

- 绝不编造具体用户的微信、电话或真实全名。
- 绝不代表平台与任何人达成合作口头承诺。
- 当接口返回空值或没有完美匹配时，给出平台检索提示（例如：“建议您前往爱合伙 App 检索关键词 `XX` 寻找匹配的伙伴”）。
