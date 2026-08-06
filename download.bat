@echo off
set url="https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip"
set file="data\raw\gtsdb\TrainIJCNN2013.zip"

:loop
echo Running curl...
curl.exe --speed-time 15 --speed-limit 1000 -L -C - -o %file% %url%
if %ERRORLEVEL% equ 0 (
    echo Download complete!
    exit /b 0
)
if %ERRORLEVEL% equ 33 (
    echo Range error - file might be fully downloaded!
    exit /b 0
)
echo Download interrupted (Error: %ERRORLEVEL%), retrying in 3 seconds...
timeout /t 3 /nobreak >nul
goto loop
