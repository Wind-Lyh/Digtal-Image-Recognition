@echo off
setlocal
set QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0PySide6\plugins\platforms
"%~dp0PanoramaStitcher.exe"
endlocal
