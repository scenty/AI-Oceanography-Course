# GitHub Pages 部署故障排除

## HttpError: Not Found 错误

如果遇到 `HttpError: Not Found` 错误，请按照以下步骤检查：

### 1. 检查 GitHub Pages 设置

**重要：这是最常见的原因！**

1. 进入您的 GitHub 仓库
2. 点击 **Settings**（设置）
3. 在左侧菜单中找到 **Pages**（页面）
4. 确保以下设置正确：
   - **Source**: 必须选择 **GitHub Actions**（不是 Branch）
   - 如果显示 "Your site is ready to be published"，说明还没有启用 GitHub Actions 作为源

### 2. 检查仓库权限

确保您的仓库允许 GitHub Actions 运行：

1. 进入 **Settings** > **Actions** > **General**
2. 在 **Workflow permissions** 部分：
   - 选择 **Read and write permissions**
   - 勾选 **Allow GitHub Actions to create and approve pull requests**

### 3. 检查分支名称

确保工作流文件中的分支名称与您的默认分支一致：

- 如果默认分支是 `main`，工作流中应该是 `- main`
- 如果默认分支是 `master`，需要修改 `.github/workflows/deploy.yml` 第 6 行

### 4. 手动触发部署测试

1. 进入 **Actions** 标签
2. 选择 **Deploy to GitHub Pages** 工作流
3. 点击 **Run workflow** 手动触发
4. 查看详细的错误信息

### 5. 检查构建输出路径

确保 `app/dist` 目录存在且包含构建文件：

```bash
cd app
npm run build
ls -la dist
```

应该能看到 `index.html` 和其他资源文件。

### 6. 检查仓库类型

- **公开仓库**：GitHub Pages 完全免费
- **私有仓库**：需要 GitHub Pro/Team/Enterprise 账户才能使用 GitHub Pages

### 7. 清除并重新部署

如果以上都正确，尝试：

1. 删除 `.github/workflows/deploy.yml` 文件
2. 重新创建并提交
3. 或者修改工作流文件（添加一个空格）并重新提交以触发新的运行

### 8. 查看详细日志

在 GitHub Actions 运行页面：

1. 点击失败的运行
2. 展开每个步骤查看详细日志
3. 查找具体的错误信息

### 常见错误信息对照

| 错误信息 | 可能原因 | 解决方法 |
|---------|---------|---------|
| `HttpError: Not Found` | GitHub Pages 未启用或源设置错误 | 在 Settings > Pages 中选择 GitHub Actions |
| `Permission denied` | 工作流权限不足 | 检查 Actions 权限设置 |
| `No such file or directory` | 构建路径错误 | 检查 `path: './app/dist'` 是否正确 |
| `npm ERR!` | 依赖安装失败 | 检查 package.json 和 package-lock.json |

### 仍然无法解决？

如果以上步骤都无法解决问题，请：

1. 检查 GitHub Actions 运行日志的完整错误信息
2. 确认仓库名称和路径配置是否正确
3. 尝试使用 GitHub CLI 或手动上传 dist 文件夹到 gh-pages 分支（传统方法）

