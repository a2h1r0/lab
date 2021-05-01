adb pull /storage/self/primary/Android/data/com.example.getpulse/files ./
adb shell rm -rf /storage/self/primary/Android/data/com.example.getpulse/files
move .\files\* .\
rmdir .\files