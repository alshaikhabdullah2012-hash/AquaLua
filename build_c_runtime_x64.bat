@echo off
echo Building AquaLua C Runtime for x64...

REM Use x64 compiler explicitly
cl /LD /O2 /DNDEBUG aqualua_runtime.c /Fe:aqualua_runtime.dll

if %ERRORLEVEL% EQU 0 (
    echo Runtime built successfully for x64!
    echo aqualua_runtime.dll is ready for distribution
) else (
    echo Build failed - make sure you're using x64 Native Tools Command Prompt
)

pause