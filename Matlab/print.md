# print

### print

**print函数可以把函数图形保存成图片：**

```matlab
minbnd = -4*pi;
maxbnd = 4*pi;
t = minbnd:0.1*pi:maxbnd;
plot(t, sin(t), 'g', 'Linewidth', 2);
line([minbnd, maxbnd], [0, 0]);    %绘制x轴  
axis([-10, 10, -2, 2])             %定义显示的坐标区间:x在(-10,10)之间，y在(-2,2)之间  
grid on;
title('sin(x)');
xlabel('x');
ylabel('sin(x)');
print('-dpng','sin.png');          %保存为png图片，在Matlab当前的工作目录下  
```

print('-dpng', 'sin.png')表示保存为png图片，文件名为sin.png，其中第一个参数可以是：

1. -dbmp：保存为bmp格式

2. -djpeg：保存为jpeg格式

3. -dpng：保存为png格式

4. -dpcx：保存为pcx格式

5. -dpdf：保存为pdf格式

6. -dtiff：保存为tiff格式

### fprintf

**fprintf(fid, format, data)中的fid表示由fopen函数打开的文件句柄，如果fid省略，则直接输出在屏幕上，format是字符串形式的输出格式，data是要输出的数据。其中format可以为：**

1. %c    单个字符
2. %d    有符号十进制数（%i也可以）
3. %u    无符号十进制数  
4. %f    浮点数（%8.4f表示对浮点数取8位宽度，同时4位小数）
5. %o    无符号八进制数
6. %s    字符串
7. %x    小写a-f的十六进制数
8. %X    大小a-f的十六进制数

```matlab
data = [5, 1, 2; 3, 7, 4];  
[row, col] = size(data);  
for i=1:row  
  for j=1:col  
    fprintf('data(%d, %d) = %d\n', i, j, data(i, j));   %直接输出到屏幕；类似于C语言的输出格式  
  end  
end
```

```matlab
%输出到文件：

data = [5, 1, 2; 3, 7, 4];
[row, col] = size(data);               %求出矩阵data的行数和列数  
  
%加t表示按Windows格式输出换行，即0xOD 0x0A，没有t表示按Linux格式输出换行，即0x0A  
fid=fopen('test.txt', 'wt');           %打开文件  
  
for i=1:row  
  for j=1:col  
    fprintf(fid, '%d ', data(i, j));   %类似于C语言的输出格式  
  end  
  fprintf(fid, '\n');  
end  
fprintf(fid, 'This is a string\n');  
fprintf(fid, '%X', hex2dec('ABCD'));  
fclose(fid);                           %最后不要忘记关闭文件！  

%就会在Matlab当前的工作目录下生成test.txt文件
```

### fscanf

从文件中读取：

我们可以使用fscanf函数

```matlab
%加t的理由和上面一样
fid=fopen('test.txt', 'rt');

%把数据读到data中。其中data是2*3的矩阵
data=fscanf(fid, '%d', [2, 3]);

s=fscanf(fid, '%s');
  
d=fscanf(fid, '%X');

%关闭文件
fclose(fid);

disp(data);
disp(s);
disp(d);

```

### disp

disp函数直接将内容输出在Matlab命令窗口中：

```matlab
!%单字符串输出：
disp('Hello World!');

%不同类型数据输出：
num1 = 1;
num2 = 2;
disp([ num2str(num1), ' + ', num2str(num2), ' = ', num2str(num1+num2)]);

%输出：
%Hello World!
```

