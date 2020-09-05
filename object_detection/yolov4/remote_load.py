# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:04:29 2020

@author: ppike
"""


import paramiko,tempfile,os



def remote_load(filename, remote_path='../home/paul/Computer_vision_new/test_images/', local_path='.', hostname='46.101.102.251',username='root',password='KAisensdata.2020Kaisens'):
    
    # Connect to server
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, username=username, password=password)
    
    
    stdin,stdout,stderr=ssh_client.exec_command('cd ../home/paul/weights_files/age_gender_estimation_weights/;ls')
     #stdin,stdout,stderr=ssh_client.exec_command('pwd')
    print('Current directory',stdout.readlines())
    
    # Load File
    src_path = remote_path+filename
    #dst_path = os.path.join(tempfile.gettempdir(),filename)
    dst_path = os.path.join(local_path,filename)
    
    ftp_client=ssh_client.open_sftp()
    ftp_client.get(src_path,dst_path)
    ftp_client.close()
    
    return dst_path
    

#remote_load('fdb40289e4e0f9031e8d93123ab40e75.5.jpeg')
remote_load('weights.28-3.73.hdf5','../home/paul/weights_files/age_gender_estimation_weights/')
    
# =============================================================================
# ssh_client = paramiko.SSHClient()
# ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh_client.connect(hostname='46.101.102.251',username='root',password='KAisensdata.2020Kaisens')
# 
# stdin,stdout,stderr=ssh_client.exec_command('cd /home;pwd')
# #stdin,stdout,stderr=ssh_client.exec_command('pwd')
# print('Current directory',stdout.readlines())
# 
# stdin,stdout,stderr=ssh_client.exec_command('pwd')
# print('Current directory',stdout.readlines())
# 
# ftp_client=ssh_client.open_sftp()
# ftp_client.get('../home/paul/Computer_vision_new/test_images/fdb40289e4e0f9031e8d93123ab40e75.5.jpeg',os.path.join(tempfile.gettempdir(),'fdb40289e4e0f9031e8d93123ab40e75.5.jpeg'))
# ftp_client.close()
# =============================================================================
