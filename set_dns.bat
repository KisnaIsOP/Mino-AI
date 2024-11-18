@echo off
echo Setting Google DNS servers...
netsh interface ip set dns "Ethernet 2" static 8.8.8.8
netsh interface ip add dns "Ethernet 2" 8.8.4.4 index=2
ipconfig /flushdns
echo DNS configuration complete.
pause
