%%  BP神经网络算法（两输入两隐含单元一输出） 
%   解决异或问题 
%   蒋晨之 2016\10
%
%   初始权值为随机生成的【-1，1】的实数，学习速率及训练误差精度由用户设定
%   w1为输入层权值矩阵，b1为其偏置系数，w2为输出层权值矩阵，b2为其偏置系数
%   n1为隐含层输入值，n2为输出层输入值，y1为隐含层输出值，y2为输出层输出值

clear
clc
%%      网络初始化========================================================
p=[0 1 0 1;1 0 0 1;1 1 0 0]; 
X=p(1:2,:);
[l,c]=size(X); 
disp('初始权值随机生成如下：');
kk=0;
presion=input('请输入训练误差精度：');
speed1=input('请输入学习速率：');
%  for ll=1:20
%  y2=0;p0=0;p1=0;p2=0;p3=0;p4=0;    
w1=rands(2,2)                               %隐含层权值初始化
w2=rands(1,2)                               %输出层权值初始化
b1=rands(2,1)
b2=rands(1)
maxstep=40000;                              %最大迭代次数
k=1;                                        %初始化迭代次数 
e=1;                                        %初始化误差
speeed2=speed1;

%%       训练BP网络=======================================================
while(e>presion&&k<maxstep)              %小于误差精度及最大迭代次数
    e=0; 
    for i=c*(k-1)+1:k*c 
        
        %%  前向搭建-----------------------------------
        %第一层
        x0=X(:,i-c*k+c);
        n1=w1*x0+b1;  
        y1=logsig(n1);   
        %第二层
        n2=w2*y1+b2; 
        y2(i)=logsig(n2);  
        
        %%  反馈算法------------------------------------
        e=e+(p(3,i-c*k+c)-y2(i))^2; 
        deta2=-2*dlogsig(n2,y2(i))*(p(3,i-c*k+c)-y2(i));    %计算输出层的deta2
        temp=zeros(size(y1,1)); 
        for j=1:size(y1,1)
            temp(j,j)=(1-y1(j))*y1(j); 
        end 
        deta1=temp*w2'*deta2;                                %计算输入层的deta1
        %权值迭代 
        w1=w1-speed1*deta1*x0';
        w2=w2-speeed2*deta2*y1'; 
        b1=b1-speed1*deta1; 
        b2=b2-speeed2*deta2; 
        
    end
    E(k)=0.5*e;
    k=k+1;
end

%%      结果输出========================================================
 for n=1:1:k-1 
     p0(n)=y2(c*n-3); 
     p1(n)=y2(c*n-2); 
     p2(n)=y2(c*n-1); 
     p3(n)=y2(c*n); 
 end 
 if k<35000 
     kk=kk+1;
 end
disp('理想输出为：1 1 0 0')
fprintf('实际输出为：%f,%f,%f,%f\n',p0(n),p1(n),p2(n),p3(n))
fprintf('最终迭代误差为：%f\n',e)
fprintf('迭代次数为：%d\n',k)
% fprintf('一共进行了%d次测试，实际输出在误差范围内的有%d次\n',ll,kk); 
%%       绘图==============================================
%  if k<35000
% 输出值变化曲线
 figure;                  
 plot(p0); 
 hold on 
 plot(p1,'r'); 
 hold on 
 plot(p2,'g'); 
 hold on 
 plot(p3,'m'); 
 legend('f(p0)','f(p1)','f(p2)','f(p3)',4) 
 title('输出值变化曲线');
%end
%end
 %误差
 figure;
 plot(E);
 legend('E',1);
 title('迭代误差变化曲线');


