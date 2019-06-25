import paramiko
ssh_client=paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname='114.115.219.202',username='root', port=20050,password='12345678')

#Downloading a file from remote machine

#ftp_client=ssh_client.open_sftp()
#ftp_client.get('remotefileth','localfilepath')
#ftp_client.close()

#Uploading file from local to remote machine

ftp_client=ssh_client.open_sftp()
ftp_client.put('/home/brian/Desktop/upload_demo','./upload_demo')
ftp_client.close()
