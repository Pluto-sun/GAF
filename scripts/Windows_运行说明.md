# Windows 脚本运行说明

本文档说明如何在Windows系统下运行增强版双路GAF网络实验脚本。

## 🎯 脚本文件说明

我们提供了两个版本的Windows脚本：

### 1. 批处理脚本 (推荐新手使用)
**文件名:** `DualGAF_enhance_dilated.bat`
- **兼容性:** 所有Windows版本
- **优点:** 简单易用，双击即可运行
- **缺点:** 功能相对简单

### 2. PowerShell脚本 (推荐高级用户)
**文件名:** `DualGAF_enhance_dilated.ps1`
- **兼容性:** Windows 7+ (预装PowerShell)
- **优点:** 功能强大，语法更现代
- **缺点:** 可能需要调整执行策略

## 🚀 运行方法

### 方法1: 运行批处理脚本
1. 打开命令提示符 (cmd)
2. 切换到脚本所在目录：
   ```cmd
   cd /d "你的项目路径\scripts"
   ```
3. 直接运行：
   ```cmd
   DualGAF_enhance_dilated.bat
   ```
   或者直接双击 `DualGAF_enhance_dilated.bat` 文件

### 方法2: 运行PowerShell脚本
1. 打开PowerShell（以管理员身份运行）
2. 如果首次运行，可能需要设置执行策略：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. 切换到脚本所在目录：
   ```powershell
   cd "你的项目路径\scripts"
   ```
4. 运行脚本：
   ```powershell
   .\DualGAF_enhance_dilated.ps1
   ```

## 📋 主要变化对比

| 元素 | Linux/Bash | Windows/Batch | Windows/PowerShell |
|------|------------|---------------|-------------------|
| 脚本标识 | `#!/bin/bash` | `@echo off` | `# PowerShell` |
| 注释 | `#` | `REM` | `#` |
| 变量定义 | `var=value` | `set var=value` | `$var = "value"` |
| 变量使用 | `$var` | `%var%` | `$var` |
| 环境变量 | `export VAR=value` | `set VAR=value` | `$env:VAR = "value"` |
| 换行符 | `\` | `^` | `` ` `` |
| 输出 | `echo` | `echo` | `Write-Host` |
| 暂停 | 无 | `pause` | `Read-Host` |

## ⚙️ 环境配置

### 1. 修改Conda环境名称
在脚本中找到以下行并修改为你的环境名：

**批处理脚本:**
```batch
call conda activate test_env
```

**PowerShell脚本:**
```powershell
& conda activate test_env
```

### 2. 修改CUDA设备
如果需要使用不同的GPU，修改：
```batch
set CUDA_VISIBLE_DEVICES=1
```
或
```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
```

### 3. 修改数据路径
根据您的实际数据路径修改：
```batch
set root_path=./dataset/DDAHU/direct_5_working
```
或
```powershell
$root_path = "./dataset/DDAHU/direct_5_working"
```

## 🔧 故障排除

### 问题1: "python不是内部或外部命令"
**解决方案:**
1. 确保Python已安装并添加到PATH
2. 或使用完整路径：`C:\path\to\python.exe run.py`

### 问题2: "conda不是内部或外部命令"
**解决方案:**
1. 确保Anaconda/Miniconda已安装
2. 重新初始化conda：
   ```cmd
   conda init cmd.exe
   ```
   或
   ```powershell
   conda init powershell
   ```

### 问题3: PowerShell执行策略限制
**解决方案:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题4: 路径包含空格
**解决方案:**
使用引号包围路径：
```batch
set root_path="./dataset/DDAHU/direct 5 working"
```

## 📝 注意事项

1. **字符编码:** 如果脚本中包含中文字符出现乱码，请确保：
   - 批处理文件保存为GBK或UTF-8编码
   - PowerShell文件保存为UTF-8编码

2. **路径分隔符:** Windows使用`\`作为路径分隔符，但在大多数情况下`/`也可以使用

3. **权限问题:** 如果遇到权限问题，请以管理员身份运行命令提示符或PowerShell

4. **环境变量:** 环境变量的设置在当前会话中有效，重新打开终端后需要重新设置

## 🎯 推荐用法

- **新手用户:** 使用批处理脚本 (`.bat`)，双击运行即可
- **高级用户:** 使用PowerShell脚本 (`.ps1`)，功能更强大
- **开发者:** 建议使用PowerShell脚本，便于调试和扩展

运行完成后，结果将保存在 `./result/` 目录下。 