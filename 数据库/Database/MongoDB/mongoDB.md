# mongoDB

**输入命令，来启动MongoDB 服务；**

```shell
mongod --dbpath D:\Mongodb\Server\data\db
```

**经过将mongoDB配置到service后（需要使用管理员权限）**

```shell
mongod -dbpath "path\data\db" -logpath "path\data\log\mongo.log" -install -serviceName "MongoDB"
```

**启动 MongoDB 命令**为：`net start MongoDB`

**关闭 MongoDB 命令**为：`net stop MongoDB`

