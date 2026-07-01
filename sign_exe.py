import os
import subprocess
import sys

def main():
    # 动态获取当前脚本所在目录作为根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 强制转换为绝对路径
    exe_path = os.path.abspath(os.path.join(root_dir, "dist", "PanoramaStitcher", "PanoramaStitcher.exe"))
    pfx_path = os.path.abspath(os.path.join(root_dir, "Lyh_Software.pfx"))
    
    print(f"[*] 检查目标程序是否存在...")
    print(f"路径: {exe_path}")
    if not os.path.exists(exe_path):
        print("[!] 错误: 找不到目标 EXE 程序！请确认已执行 PyInstaller 打包。")
        return

    print("\n[*] 开始执行自动化签名流程...")
    
    # 构建 PowerShell 签发与注入脚本
    # 采用 PowerShell 方案以脱离对 Windows SDK (signtool.exe) 的强依赖
    ps_script = f"""
    $ErrorActionPreference = 'Stop'
    
    Write-Host "[1/3] 正在生成 Lyh_Software 代码签名证书..."
    $cert = New-SelfSignedCertificate -Subject "CN=Lyh_Software" -Type CodeSigningCert -CertStoreLocation "Cert:\\CurrentUser\\My"
    
    Write-Host "[2/3] 正在导出证书备份 ( Lyh_Software.pfx )..."
    $pwd = ConvertTo-SecureString -String "123456" -Force -AsPlainText
    Export-PfxCertificate -Cert $cert -FilePath "{pfx_path}" -Password $pwd | Out-Null
    
    Write-Host "[3/3] 正在将证书强行注入到 EXE 中..."
    $signResult = Set-AuthenticodeSignature -FilePath "{exe_path}" -Certificate $cert
    
    Write-Host "==========================="
    Write-Host "验证签名状态:"
    if ($signResult.Status -eq 'Valid') {{
        Write-Host "✅ 签名成功！状态: Valid" -ForegroundColor Green
    }} else {{
        Write-Host "❌ 签名未通过校验！状态: $($signResult.Status)" -ForegroundColor Red
        Write-Host "详细信息: $($signResult.StatusMessage)" -ForegroundColor Red
    }}
    
    Write-Host "==========================="
    Write-Host "如需手动进行标准验证，请在 PowerShell 中执行以下命令:"
    Write-Host "Get-AuthenticodeSignature -FilePath '{exe_path}'"
    """

    print("[*] 正在调用系统 PowerShell 引擎...")
    try:
        # 调用 PowerShell 并绕过执行策略
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script], 
            capture_output=True, 
            text=True,
            encoding='gbk'
        )
        
        print("\n=== PowerShell 运行日志 ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== PowerShell 异常信息 ===")
            print(result.stderr)
            
    except Exception as e:
        print(f"[!] 调用 PowerShell 时发生未知异常: {e}")

if __name__ == "__main__":
    main()
