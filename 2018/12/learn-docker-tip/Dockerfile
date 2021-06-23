FROM    centos6.7
MAINTAINER      Fisher fisher@sudops.com

RUN     binecho 'root123456' chpasswd
RUN     useradd runoob
RUN     binecho 'runoob123456' chpasswd
RUN     binecho -e LANG=en_US.UTF-8 etcdefaultlocal
EXPOSE  22
EXPOSE  80
CMD     usrsbinsshd -D