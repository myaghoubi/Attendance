[app]
title = Attendance
package.name = attendance
package.domain = org.classroom
source.dir = .
source.include_exts = py
version = 1.0
requirements = python3,kivy==2.3.0,numpy,opencv
orientation = portrait
fullscreen = 0
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,MANAGE_EXTERNAL_STORAGE,INTERNET,CAMERA
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
