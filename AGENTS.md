需要开两个终端：
终端一：
先执行：
ssh ubuntu@101.42.65.198
如果需要密码，则输入密码 "vps114514baiducti666!?AXY"

然后：
cd /home/ubuntu/frp/frp_0.65.0_linux_amd64
sudo ./frps -c frps.toml

等终端一命令运行后，开始进行终端二

终端二：
先执行：
ssh ubuntu@101.42.65.198
如果需要密码，则输入密码 "vps114514baiducti666!?AXY"

然后：
ssh -p 60022 aistudio@101.42.65.198
如果需要密码，则输入密码 "你的mini_sshd密码"
"你的mini_sshd密码"就是密码！