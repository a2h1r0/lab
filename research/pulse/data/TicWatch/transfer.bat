adb connect 192.168.11.64:5555
adb -s 192.168.11.64:5555 pull /storage/self/primary/Android/data/com.example.getpulse/files ./
adb -s 192.168.11.64:5555 shell rm -rf /storage/self/primary/Android/data/com.example.getpulse/files
move .\files\* .\
rmdir .\files