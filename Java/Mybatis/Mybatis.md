# Mybatis

## @Param注解

**如下几个需要添加 @Param 注解的场景**

**第一种：方法有多个参数，需要 @Param 注解**

```java
@Mapper
public interface UserMapper {
 Integer insert(@Param("username") String username, @Param("address") String address);
}
```

对应的 XML 文件如下：

```xml
<insert id="insert" parameterType="org.javaboy.helloboot.bean.User">
    insert into user (username,address) values (#{username},#{address});
</insert>
```

这是最常见的需要添加 @Param 注解的场景。

**第二种：方法参数要取别名，需要 @Param 注解**

当需要给参数取一个别名的时候，我们也需要 @Param 注解，例如方法定义如下：

```java
@Mapper
public interface UserMapper {
 User getUserByUsername(@Param("name") String username);
}
```

对应的 XML 定义如下：

```xml
<select id="getUserByUsername" parameterType="org.javaboy.helloboot.bean.User">
    select * from user where username=#{name};
</select>
```

**第三种：XML 中的 SQL 使用了 $ ，那么参数中也需要 @Param 注解**

$会有注入漏洞的问题，但是有的时候不得不使用$符号，例如要传入列名或者表名的时候，这个时候必须要添加 @Param 注解，例如：

```java
@Mapper
public interface UserMapper {
 List<User> getAllUsers(@Param("order_by")String order_by);
}
```

对应的 XML 定义如下：

```xml
<select id="getAllUsers" resultType="org.javaboy.helloboot.bean.User">
    select * from user
 	<if test="order_by!=null and order_by!=''">
        order by ${order_by} desc   <!--order by此处只能使用order_by-->
 	</if>
</select>
```

**第四种，那就是动态 SQL ，如果在动态 SQL 中使用了参数作为变量，那么也需要 @Param 注解，即使你只有一个参数。**

**如果我们在动态 SQL 中用到了 参数作为判断条件，那么也是一定要加 @Param 注解的**，例如如下方法：

```java
@Mapper
public interface UserMapper {
 List<User> getUserById(@Param("id")Integer id);
}
```

定义出来的 SQL 如下：

```xml
<select id="getUserById" resultType="org.javaboy.helloboot.bean.User">
    select * from user
 	<if test="id!=null">
        where id=#{id}
 	</if>
</select>
```

## resultType和resultMap

**注意insert、update、 delete元素均没有resultType属性,返回int表示成功条数。**

在 MyBatis 中，`resultType` 和 `resultMap` 都是用于指定查询结果的映射方式，但它们有一些区别。

- `resultType`：`resultType` 用于指定查询结果的类型，通常是一个具体的 Java 类型。这个类型应该与查询结果中的列名一一对应，MyBatis 将使用反射机制来创建一个该类型的对象，并将查询结果映射到该对象的属性上。每个列名将与对象的属性名进行匹配，通过 setter 方法将值设置到对应的属性中。例如：

```xml
<select id="getUser" resultType="org.example.User">
  SELECT id, username, email FROM users WHERE id = #{id}
</select>
```

在这个例子中，`resultType` 指定了查询结果应映射到 `org.example.User` 类型的对象。查询的结果集中的列 `id`、`username` 和 `email` 会通过对应的 setter 方法设置到 `User` 对象的属性上。

- `resultMap`：`resultMap` 是一种更为灵活和可定制化的结果映射方式。它允许你根据需要自定义映射规则，将查询结果映射到一个复杂的对象结构中。`resultMap` 定义了如何映射查询结果的列和 Java 对象的属性之间的关系。你可以指定列与属性的对应关系、类型转换器、关联关系等。

```xml
<resultMap id="userResultMap" type="org.example.User">
  <id property="id" column="user_id"/>
  <result property="username" column="user_name"/>
  <result property="email" column="user_email"/>
</resultMap>

<!--property:User的实体属性  对应  colume:数据库的表项名称（列名）-->

<select id="getUser" resultMap="userResultMap">
  SELECT user_id, user_name, user_email FROM users WHERE id = #{id}
</select>
```

在上述示例中，我们定义了一个名为 `userResultMap` 的 `resultMap`，并指定了列与属性的映射关系。查询语句中的列 `user_id`、`user_name` 和 `user_email` 会按照映射规则，通过反射或类型转换器等机制设置到 `User` 对象的属性中。

总结来说，`resultType` 适用于简单的映射情况，只需将查询结果映射到一个具体类型的对象。而 `resultMap` 则提供了更大的灵活性和定制性，可以处理更复杂的映射需求，允许自定义映射规则和处理关联关系。

**二者不能同时存在，本质上都是Map的数据结构**

### resultMap

**resultMap属性**
id : resultMap的唯一标识
type : Java实体类

**resultMap子元素**
id
一般对应数据库中该行的主键id，设置此项可提高MyBatis性能
result
映射到JavaBean的某个“简单类型”属性
association
映射到JavaBean的某个“复杂类型”属性（一对一），比如JavaBean类
collection
映射到JavaBean的某个“复杂类型”属性（一对多），比如集合

**association**

```xml
   <!--高级映射一对一-->
    <resultMap id="studentGradeVo" type="StudentGradeVo">
        <id property="student.studentId" column="student_id"></id>
        <!--学生是一-->
        <association property="student" javaType="Student">
            <id property="studentId" column="student_id"></id>
            <result property="gradeId" column="grade_id"></result>
            <result property="studentAddress" column="student_address"></result>
            <result property="studentName" column="student_name"></result>
            <result property="studentBirth" column="student_birth"></result>
            <result property="studentNo" column="student_no"></result>
            <result property="studentAge" column="student_age"></result>
        </association>
        <!--一个学生对应一个年级-->
        <association property="grade" javaType="Grade">
            <id property="gradeId" column="grade_id"></id>
            <result property="gradeName" column="grade_name"></result>
        </association>
    </resultMap>


    <select id="getStudentGradeVoByStudentId" resultMap="studentGradeVo">
        select * from student, grade
        where student.grade_id = grade.grade_id
          and student.student_id = #{studentId}
    </select>
```

**说明**

1. javaType="Student"  需要配置扫描实体类

   ```xml
   <typeAliases>
       <!--配置一个具体的类-->
       <!--<typeAlias type="com.scu.mybatis.pojo.Student" alias="student"></typeAlias>-->
       <!--扫描实体类的包-->
       <package name="com.example.template.pojo"></package>
       <package name="com.example.template.vo"></package>
   </typeAliases>
   ```

2. \<id property="student.studentId" column="student_id">\</id>

   **此处的property需要配置的是studentGradeVo的具体属性**

3. ```xml
   	<association property="student" javaType="Student">
               <id property="studentId" column="student_id"></id>
               <result property="gradeId" column="grade_id"></result>
               <result property="studentAddress" column="student_address"></result>
               <result property="studentName" column="student_name"></result>
               <result property="studentBirth" column="student_birth"></result>
               <result property="studentNo" column="student_no"></result>
               <result property="studentAge" column="student_age"></result>
   	</association>
   ```
    **此处的property因为再studentproperty下，所以不需要配置具体属性**

**collection**

```xml
<!--高级映射一对多-->
<resultMap id="gradeStudentVo" type="GradeStudentVo">
    <id property="grade.gradeId" column="grade_id"></id>
    <!--年级是一-->
    <association property="grade" javaType="Grade">
        <id property="gradeId" column="grade_id"></id>
        <result property="gradeName" column="grade_name"></result>
    </association>
    <!--一个年级多个学生-学生是多-->
    <collection property="students" ofType="Student">
        <id property="studentId" column="student_id"></id>
        <result property="gradeId" column="grade_id"></result>
        <result property="studentAddress" column="student_address"></result>
        <result property="studentName" column="student_name"></result>
        <result property="studentBirth" column="student_birth"></result>
        <result property="studentNo" column="student_no"></result>
        <result property="studentAge" column="student_age"></result>
    </collection>
</resultMap>
```

## Mybatis接收的参数类型

1. 基本类型   key:变量名  value:变量值
2. 对象    key:对象的属性名   value:对象的属性值
3. List
4. 数组(Array)
5. Map

**无论是入参是何种参数类型，都会把其放在一个Map中**

## Mybatis的自动匹配机制

1. ```xml
   <association property="student" javaType="Student">
       <id property="studentId" column="student_id"></id>
       <result property="gradeId" column="grade_id"></result>
       <result property="studentAddress" column="student_address"></result>
       <result property="studentName" column="student_name"></result>
       <result property="studentBirth" column="student_birth"></result>
       <result property="studentNo" column="student_no"></result>
       <result property="studentAge" column="student_age"></result>
   </association>
   ```

2. ```xml
   <select id="getStudentById" resultType="Student">
       select * from student where student_id=#{studentId};
   </select>
   ```

**Student类为了区别，专门写成了如下定义**

```java
package com.example.template.pojo;

public class Student {
    long studentI;
    String studentName;
    String studentBirth;
    String studentAddress;
    long studentNo;
    int gradeId;
    int studentAge;

    public Student() {
    }

    public Student(long studentId, String studentName, String studentBirth, String studentAddress, long studentNo, int gradeId, int studentAge) {
        this.studentI = studentId;
        this.studentName = studentName;
        this.studentBirth = studentBirth;
        this.studentAddress = studentAddress;
        this.studentNo = studentNo;
        this.gradeId = gradeId;
        this.studentAge = studentAge;
    }

    public long getStudentId() {
        return studentI;
    }

    public void setStudentId(long studentId) {
        this.studentI = studentId;
    }

}
```

**本质上是调用了set方法去匹配的**（property="studentId"，和select方法返回的Student都是通过set方法去匹配的）

**说明：**

发现只要不改变`setStudentId`函数，就能完成匹配

而改变`setStudentId`函数的名称，就不能完成匹配

### resultMap自动映射匹配

**字段名与属性名一致,或者字段单词是下划线隔开,属性是驼峰,不用在setting中显式的声明驼峰**

### select ORM对应

```xml
  <settings>
        <setting name="autoMappingBehavior" value="PARTIAL"/>
        <setting name="mapUnderscoreToCamelCase" value="true"/>   <!--这句话需要写-->
  </settings>
```

## Mybatis动态SQL

**定义**

```xml
<sql id="fromStudent">
  select * from student
</sql>
```

**引用**

```xml
<select id="getStudentGradeVoByCondition" resultMap="studentGradeVo2">
  <include refid="fromStudent"></include> where student_name like concat('%', #{studentName} ,'%');
</select>
```
### 1.使用动态SQL完成多条件查询

**where-if**


注意:**where 标签**中只会去除多余的and 并不会加上and

Dao
```java
List<StudentGradeVo> getStudentGradeVoByConditionWhereIf(Student student);
```
Mapper
```java
<!--动态sql where if-->
<select id="getStudentGradeVoByConditionWhereIf" resultMap="studentGradeVo2">
  <include refid="fromStudent"></include>
  <where>
    <if test="studentName neq null and studentName neq ''">
      and student_name like concat('%', #{studentName} ,'%')
    </if>
    <if test="studentAddress neq null and studentAddress neq ''">
      and student_address like concat('%', #{studentAddress} ,'%')
    </if>
    <if test="studentNo neq null and studentNo neq 0">
      and student_no = #{studentNo}
    </if>
  </where>
</select>

```

Test
```java
@Test
public void test(){
  SqlSession sqlSession = sqlSessionFactory.openSession();
  StudentDao mapper = sqlSession.getMapper(StudentDao.class);
  //封装参数
  Student student = new Student();
  student.setStudentName("小");
  student.setStudentAddress("道");
  //调用方法
  List<StudentGradeVo> studentGradeVos = mapper.getStudentGradeVoByConditionWhereIf(student);
  studentGradeVos.forEach(studentGradeVo -> {
    System.out.println(studentGradeVo.getGrade().getGradeName() + studentGradeVo.getStudent());
  });
  sqlSession.close();
}
```


注意:

单个基本数据类型的if test 判断写法

```xml
<if test="_parameter !=0 ">AND p.id=#{pId}</if>
```

**_parameter代表单个入参基本数据类型的通用变量名**

### 2.使用动态SQL实现更新操作

### 3.使用foreach完成复杂查询

**相当于对sql进行for操作**

### 4.choose

**类似于swith关键字**

### 5.注解@开发动态sql

### 6.MyBatis实现分页功能