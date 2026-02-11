# GitHub Pages 部署说明

本项目已配置好 GitHub Pages 自动部署功能。

## 部署步骤

### 1. 启用 GitHub Pages

1. 进入您的 GitHub 仓库
2. 点击 **Settings**（设置）
3. 在左侧菜单中找到 **Pages**（页面）
4. 在 **Source**（源）部分，选择：
   - **Source**: `GitHub Actions`
5. 保存设置

### 2. 配置仓库名称（重要）

如果您的 GitHub 仓库名称不是 `AI-Oceanography-Course`，请修改 `app/vite.config.ts` 文件中的 `base` 路径：

```typescript
base: process.env.NODE_ENV === 'production' ? '/您的仓库名/' : './',
```

例如，如果仓库名是 `my-website`，则改为：
```typescript
base: process.env.NODE_ENV === 'production' ? '/my-website/' : './',
```

### 3. 推送代码

将代码推送到 `main` 分支（如果您的默认分支是 `master`，请修改 `.github/workflows/deploy.yml` 中的分支名）：

```bash
git add .
git commit -m "配置 GitHub Pages 部署"
git push origin main
```

### 4. 查看部署状态

1. 在 GitHub 仓库页面，点击 **Actions** 标签
2. 查看工作流运行状态
3. 部署完成后，您的网站将在以下地址访问：
   - `https://您的用户名.github.io/AI-Oceanography-Course/`

## 自动部署

每次您向 `main` 分支推送代码时，GitHub Actions 会自动：
1. 安装依赖
2. 构建项目
3. 部署到 GitHub Pages

## 手动触发部署

如果需要手动触发部署：
1. 进入 **Actions** 标签
2. 选择 **Deploy to GitHub Pages** 工作流
3. 点击 **Run workflow** 按钮

## 故障排除

### ❌ HttpError: Not Found 错误

**这是最常见的错误！** 通常是因为 GitHub Pages 没有正确启用。

**解决方法：**

1. **最重要的一步**：进入仓库的 **Settings** > **Pages**
2. 在 **Source** 部分，**必须选择 `GitHub Actions`**（不要选择 Branch）
3. 如果显示 "Your site is ready to be published"，说明还没有选择源，请选择 **GitHub Actions**
4. 保存后，重新运行工作流

**其他可能的原因：**

- 检查 **Settings** > **Actions** > **General** > **Workflow permissions**，确保选择了 **Read and write permissions**
- 确认您的默认分支名称（main 或 master）与工作流文件中的配置一致
- 详细故障排除请查看 `TROUBLESHOOTING.md` 文件

### 如果网站无法正常显示

1. 检查 `vite.config.ts` 中的 `base` 路径是否正确
2. 确保 GitHub Pages 设置中选择了 **GitHub Actions** 作为源
3. 检查 Actions 标签中的错误信息

### 如果资源文件加载失败

确保所有图片和资源文件都在 `app/public` 目录下，Vite 会自动处理这些文件。

## 自定义域名（可选）

如果您想使用自定义域名：
1. 在仓库的 **Settings** > **Pages** 中配置自定义域名
2. 将 `vite.config.ts` 中的 `base` 改为 `'/'`
3. 按照 GitHub 的说明配置 DNS 记录

