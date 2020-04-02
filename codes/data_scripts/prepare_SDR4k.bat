rem convert 10-bit HDR video to 10-bit SDR video
for %a in ("*.mp4") do ffmpeg -i "%a" ^
-vf zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p10le,zscale=s=3840x2160 ^
-c:v libx265 -preset slow -crf 18 -c:a copy "SDR_10bit\\video\\%~na.mp4"

rem convert video to image sequences
for %a in ("video\\*.mp4") do (
md imgs\\%~na
ffmpeg -i "%a" "imgs\\%~na\\%03d.png")