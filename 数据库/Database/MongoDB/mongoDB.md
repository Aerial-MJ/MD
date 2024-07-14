# mongoDB

[PostgreSQL 可以使用键值对来存储数据，而不是使用两个列表进行插入操作](https://geek-docs.com/postgresql/postgresql-questions/256_postgresql_postgresql_insert_with_keyvalue_instead_of_two_lists.html)

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