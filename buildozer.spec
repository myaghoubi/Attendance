[app]
title = Attendance
package.name = attendance
package.domain = gu.ac.ir
source.dir = .
source.include_exts = py
version = 1.0
requirements = python3==3.11.5,hostpython3==3.11.5,kivy==2.3.0,numpy
p4a.branch = v2024.01.21
orientation = portrait
fullscreen = 0
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,MANAGE_EXTERNAL_STORAGE,INTERNET,CAMERA
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.accept_sdk_license = True
android.ndk = 25b
android.ndk_api = 24

[buildozer]
log_level = 2
warn_on_root = 1
