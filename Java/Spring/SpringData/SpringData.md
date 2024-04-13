# Spring Data

## Spring Data JDBC 

Spring Data JDBC 是 Spring Data 项目家族中的一个模块，用于简化基于 JDBC（Java Database Connectivity）的数据访问。它提供了一种更简洁和类型安全的方式来处理数据库操作，同时保留了灵活性和性能。

Spring Data JDBC 的设计理念是将数据库操作转化为领域对象（Domain Objects）的操作，而不是直接操作 SQL。它支持使用简单的 Java 类来表示数据库表和字段，并通过注解或约定来进行映射。开发者可以定义实体类（Entity Class）来表示数据库表，使用注解或命名约定来映射表和字段，以及定义关联关系。

Spring Data JDBC 提供了一组 CRUD（Create-Read-Update-Delete）操作的默认实现，可以直接操作数据库表，而不需要手动编写 SQL 语句。它还支持事务管理、批量操作、分页查询等常见的数据库操作需求。

与传统的 ORM（对象关系映射）框架相比，Spring Data JDBC 更加轻量级和简单，不需要引入复杂的映射配置文件或显式的持久化方法。它遵循 Spring 的哲学，鼓励开发者使用简单的纯粹的 Java 对象进行数据库操作，并通过自动化的机制来实现与数据库的交互。

**总而言之，Spring Data JDBC 是一个基于 JDBC 的数据访问模块，旨在简化数据库操作，提供了简洁和类型安全的方式来处理数据库访问。它通过领域对象的操作，减少了编写 SQL 语句的工作量，同时保留了灵活性和性能。**

```java
package com.aerial.Controller;
 
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
 
import java.util.List;
import java.util.Map;
 
@RestController
public class JdbcController {
 
    @Autowired
    JdbcTemplate jdbcTemplate;
 
    //查询数据库的所有信息（CURD）
    //没有实体类，我们怎么获取数据库的东西
    @GetMapping("/userList")
    public List<Map<String, Object>> userList() {
        //直接写到数据库中，所以不用Model传值
        String sql = "select * from user";
        List<Map<String, Object>> maps = jdbcTemplate.queryForList(sql);
        return maps;
    }
 
    @GetMapping("/addUser")
    public void addUser() {
        //直接写到数据库中，所以不用Model传值
        String sql = "insert into user(userId,user_name,password) value (4,'xiaoming','123456')";
        jdbcTemplate.update(sql);
        //增删改都是update
    }
 
    @RequestMapping("/updateUser/{id}")
    public void updateUser(@PathVariable("id") Integer id) {
        //直接写到数据库中，所以不用Model传值
        //sql语句
        String sql = "update user set user_name=?,password=? where userId=" + id;
 
        //传值
        Object[] objects = new Object[2];
        objects[0] = "xiaoming2";
        objects[1] = "zzzzzz";
        jdbcTemplate.update(sql, objects);
        //增删改都是update
    }
 
    @GetMapping("/deleteUser/{id}")
    public void deleteUser(@PathVariable("id") Integer id) {
        //直接写到数据库中，所以不用Model传值
        String sql = "delete from user where userId=?";
        jdbcTemplate.update(sql,id);
        //增删改都是update
    }
 
}
```

## 其余Data项目

除了 Spring Data JDBC，Spring Data 项目家族还包括以下模块：

1. Spring Data JPA：提供了对基于 JPA（Java Persistence API）的数据访问的支持。它简化了 JPA 的使用，通过自动生成查询和持久化方法，减少了手动编写 SQL 或 JPQL（Java Persistence Query Language）的工作量。

2. Spring Data MongoDB：提供了对 MongoDB NoSQL 数据库的支持。它提供了一组简洁的 API，用于与 MongoDB 进行交互，支持复杂查询、聚合操作和地理位置查询等功能。

3. Spring Data Redis：提供了对 Redis 键值存储的支持。它封装了 Redis 的操作，提供了简单的 API，用于操作 Redis 的字符串、哈希、列表、集合和有序集合等数据结构。

4. Spring Data Elasticsearch：提供了对 Elasticsearch 分布式搜索和分析引擎的支持。它简化了与 Elasticsearch 的交互，提供了高级查询、聚合操作和全文搜索等功能。

5. Spring Data REST：通过自动化的方式，将基于 Spring Data 的存储库（Repository）暴露为 RESTful API。它简化了构建 RESTful 服务的过程，支持对实体的增删改查操作，并提供了自动分页、排序和关联资源的支持。

6. Spring Data Neo4j：提供了对 Neo4j 图数据库的支持。它允许开发者使用领域对象和注解的方式来操作图数据，支持节点（Node）和关系（Relationship）的创建、查询和持久化。

每个 Spring Data 模块都旨在简化特定类型的数据访问，提供统一的编程模型和一组高级功能，使开发者能够更轻松地与不同类型的数据存储进行交互。这些模块可以单独使用，也可以与其他 Spring 技术和框架集成，以构建全功能的应用程序。

**mybatis不是SpringData的项目**

## 连接池和数据源

数据源（DataSource）和连接池（Connection Pool）都是在应用程序中管理和获取数据库连接的重要组件。

**数据源**（DataSource）是数据库连接的工厂，它提供了获取数据库连接的方法。数据源包含了与数据库的连接信息，如数据库的URL、用户名、密码等。在应用程序中，**通过数据源可以获取到数据库连接对象，以便进行数据库操作。**

连接池（Connection Pool）是一种**管理和复用数据库连接的技术**。它维护了一个连接的池子（Pool），在应用程序初始化阶段就创建**一定数量的数据库连接**，并将这些连接存放在池中。当应用程序需要连接数据库时，可以从**连接池中获取一个可用的连接。使用完毕后，将连接归还给连接池，而不是关闭连接。连接池负责管理连接的分配、回收和维护，以便在后续的数据库操作中可以重复利用连接，避免频繁地创建和销毁连接的开销。**

连接池的好处是可以提高数据库操作的性能和效率，主要体现在以下几个方面：

1. 连接复用：连接池可以重复利用已经创建的连接，避免了重复创建和销毁连接的开销，提高了数据库操作的效率。

2. 连接管理：连接池可以管理连接的分配和回收过程，确保连接的有效性和可靠性。它可以检测空闲连接的有效性，自动关闭空闲时间过长的连接，并创建新的连接以满足需要。

3. 连接限制：连接池可以限制应用程序对数据库的并发连接数量，避免过多的连接导致数据库性能下降或资源耗尽的问题。

4. 连接配置：连接池允许配置连接的属性，如最大连接数、最小连接数、超时时间等，以满足应用程序的需求。

通过使用连接池，应用程序可以更高效地管理和利用数据库连接资源，提升数据库操作的性能和可靠性。常见的连接池实现包括 Apache Commons DBCP、HikariCP、C3P0 等。

### Spring使用的数据源

在 Spring 框架中，提供了几个常用的数据源实现，用于与数据库进行交互。以下是一些常见的数据源：

1. DriverManagerDataSource：这是一个简单的数据源实现，使用 JDBC 的 DriverManager 获取数据库连接。它需要手动配置数据库连接信息，包括数据库的 URL、用户名和密码。

2. BasicDataSource：这是 Apache Commons DBCP（数据库连接池）的一个实现，也是 Spring 框架中最常用的数据源之一。它提供了连接池功能，可以配置最大连接数、最小连接数、连接超时时间等属性。

3. HikariDataSource：这是一个高性能的 JDBC 数据源，基于 HikariCP 连接池。HikariCP 是目前较为流行的连接池实现，具有快速启动、低资源消耗和高度可定制的特点。

4. TomcatDataSource：这是 Apache Tomcat 数据源（Tomcat JDBC Pool）的一个封装，提供了与 Tomcat 连接池的集成。它适用于在 Tomcat 容器中运行的应用程序。

这些数据源实现都可以通过配置文件或编程方式进行配置，并与 Spring 的数据访问模块（如 Spring JDBC、Spring Data JPA）进行集成。在使用时，可以根据具体的需求选择合适的数据源，并进行相应的配置。Spring 的数据源抽象层使得在应用程序中切换不同的数据源变得简单，同时提供了对连接池的管理和控制，提高了数据库操作的性能和可靠性。

### 数据源一般都是基于连接池的

大多数情况下，数据源都是基于连接池的。连接池是一种管理数据库连接的技术，它维护了一组预先创建的数据库连接，应用程序可以从连接池中获取连接并在使用完后归还给连接池，而不是每次都创建新的连接和关闭连接。

连接池的使用有以下几个主要优点：

1. 连接复用：连接池可以重复利用已经创建的连接，避免了频繁创建和销毁连接的开销，提高了数据库操作的效率。

2. 连接管理：连接池负责连接的分配和回收，可以确保连接的有效性和可靠性。它可以检测空闲连接的有效性，关闭空闲时间过长的连接，并创建新的连接以满足需要。

3. 连接限制：连接池可以限制应用程序对数据库的并发连接数量，避免过多的连接导致数据库性能下降或资源耗尽的问题。

4. 连接配置：连接池允许配置连接的属性，如最大连接数、最小连接数、超时时间等，以满足应用程序的需求。

在 Java 应用程序中，基于连接池的数据源成为了常见的选择。连接池的实现可以使用开源的**连接池库**，如 Apache Commons DBCP、HikariCP、C3P0 等，这些库提供了可靠和高性能的连接池实现，可以与 Spring 框架进行集成，方便地使用在应用程序中。

### mybatis使用的是连接池技术

MyBatis 可以与连接池技术集成，以提高数据库连接的效率和性能。

通常情况下，MyBatis 的应用程序会配置一个连接池，以管理数据库连接。连接池可以在应用程序启动时初始化一组数据库连接，并根据需要将这些连接分配给 MyBatis 的数据访问操作。当操作完成后，连接会被归还给连接池，而不是被关闭。

MyBatis 并不直接提供连接池功能，而是通过与第三方连接池库集成来实现连接池的管理。常用的连接池库包括 Apache Commons DBCP、HikariCP、C3P0 等。这些连接池库提供了高效的连接池实现，能够管理和复用数据库连接，提供连接的管理、连接的分配和回收等功能。

在 MyBatis 的配置文件中，可以配置连接池相关的属性，如连接池类型、最大连接数、最小连接数、连接超时时间等。这些属性可以根据应用程序的需求进行调整，以达到最佳的性能和资源利用率。

通过与连接池的集成，MyBatis 可以有效地管理数据库连接，避免频繁地创建和关闭连接，提高数据库操作的性能和可靠性。连接池能够根据应用程序的需求动态地管理连接的数量，并提供连接的复用，从而减少了连接的创建和销毁开销，优化了数据库访问的性能。

### sqlsession.close()函数

MyBatis 中的 `sqlSession.close()` 函数的作用就是将数据库连接归还给连接池。

在 MyBatis 中，每个数据库操作都是通过 `SqlSession` 对象来执行的。`SqlSession` 表示与数据库的一次会话，它负责管理数据库连接和执行 SQL 语句。

在完成一次数据库操作后，为了释放数据库连接资源，必须调用 `sqlSession.close()` 方法来关闭 `SqlSession` 对象。关闭 `SqlSession` 对象会将其中的数据库连接归还给连接池，以便其他的数据库操作可以复用该连接。

通过归还连接给连接池，可以有效地管理连接资源，避免了连接的频繁创建和销毁开销，提高了数据库操作的性能和可靠性。连接池会负责连接的回收和再利用，确保连接的有效性和可靠性。

需要注意的是，`SqlSession` 对象在使用完毕后必须及时关闭，否则会导致连接资源的泄漏和浪费。通常的做法是将 `SqlSession` 对象放在 `try-finally` 块中，在 `finally` 块中调用 `close()` 方法，以确保连接的正确释放。

示例代码如下所示：
```java
SqlSession sqlSession = sqlSessionFactory.openSession();
try {
    // 执行数据库操作
    // ...
} finally {
    sqlSession.close(); // 关闭 SqlSession 对象，归还数据库连接给连接池
}
```

通过正确使用 `SqlSession` 对象并在适当的时候关闭它，可以确保连接池得到有效地利用，提高数据库操作的性能和资源利用率。

### 连接池有哪些

Spring Framework 可以与多个连接池集成，以便在应用程序中管理数据库连接。以下是一些常用的连接池实现：

1. Apache Commons DBCP：Apache Commons DBCP（数据库连接池）是一个开源的、成熟的连接池实现。它提供了可配置的连接池参数，支持连接池的创建、管理和释放。

2. HikariCP：HikariCP 是一个高性能的 JDBC 连接池实现，被广泛认可为速度和资源效率方面的最佳选择。它具有轻量级的设计，以快速获取和释放连接，并提供高度可配置的选项。

3. Tomcat JDBC Pool：Tomcat JDBC Pool 是 Apache Tomcat 项目的一部分，它提供了一个可靠的 JDBC 连接池实现。它支持连接池的自动管理和监控，并具有各种配置选项。

4. C3P0：C3P0 是一个成熟的、高性能的 JDBC 连接池实现，提供了许多高级功能，如连接池的连接重试、连接泄漏检测和超时设置。

5. Druid：Druid 是一个开源的、高性能的 JDBC 连接池实现，由阿里巴巴开发并维护。它具有强大的监控和统计功能，并支持连接池的动态调整。

这些连接池实现都可以与 Spring Framework 集成，并通过配置文件或编程方式进行设置和管理。你可以根据项目需求、性能要求和个人偏好选择适合的连接池实现。