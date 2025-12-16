@echo off
echo Building AquaLua C Runtime...

REM Try to compile with cl directly (should work in Developer Command Prompt)
cl /LD /O2 /DNDEBUG aqualua_runtime.c /Fe:aqualua_runtime.dll

if %ERRORLEVEL% EQU 0 (
    echo Runtime built successfully!
    echo aqualua_runtime.dll is ready for distribution
) else (
    echo Build failed - make sure you're running from Visual Studio Developer Command Prompt
)

pause