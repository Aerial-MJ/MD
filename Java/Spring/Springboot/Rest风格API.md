**@RestController注解**

如果一个Controller类主要用于做网络服务，所有的方法都需要添加@ResponseBody注解，我们可以在类的前面加上@RestController注解，这个注解相当于@Controller+@ResponseBody，也就是说它既可以用于声明类为Controller，也可以给所有方法自动添加@ResponseBody注解。 

**@ResponseBody注解**在使用SpringMVC时，Controller中的方法返回值会通过视图处理器ViewResolver处理为页面的URL，然后跳转到对应页面中，如下面例子返回的hello，将转换为`/WEB-INF/pages/hello.jsp`

```java
@Controller
public class UserController{
	@RequestMapping(“/login”)
	public String login(){
		return “hello”;
	}
} 
```

有时候我们需要用Controller实现网络服务接口，比如：商品查询、查询天气、新闻数据等。就不需要进行页面跳转而是直接返回数据。 这时候我们可以在方法上，添加注解：@ResponseBody 

```java
@Controller
public class UserController{
	@RequestMapping(“/login”)
    @ResponseBody
	public String login(){
		return “hello”;
	}
} 
```

在浏览器中直接显示hello文字，也就是说添加了@ResponseBody注解的方法，返回值会通过HTTP响应主体直接发送给浏览器。

## rest Api 创建

在Spring Boot框架中，可以使用Spring MVC来创建REST API。以下是在Spring Boot中编写REST API的一般步骤：

1. 添加依赖：在项目的pom.xml文件中添加Spring Boot和Spring MVC的依赖。例如，使用以下依赖项：
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

2. 创建控制器类：创建一个控制器类来处理API请求和响应。在该类上使用`@RestController`注解，以指示该类是一个REST控制器。在控制器类中，可以定义各种处理HTTP请求的方法，例如GET、POST、PUT、DELETE等。

```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    // GET请求示例
    @GetMapping
    public List<User> getUsers() {
        // 处理获取用户列表的逻辑
    }

    // POST请求示例
    @PostMapping
    public User createUser(@RequestBody User user) {
        // 处理创建用户的逻辑
    }

    // PUT请求示例
    @PutMapping("/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody User user) {
        // 处理更新用户的逻辑
    }

    // DELETE请求示例
    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        // 处理删除用户的逻辑
    }
}
```

3. 处理请求和响应：在控制器方法中，可以使用Spring MVC提供的注解来处理请求参数、路径变量、请求体等。例如，使用`@PathVariable`注解获取路径中的变量，使用`@RequestBody`注解获取请求体中的数据。

4. 配置URL映射：可以使用`@RequestMapping`注解在控制器类和方法上设置URL映射路径。可以在类级别上设置基本路径，并在方法级别上设置具体的子路径。

5. 返回响应：控制器方法可以返回各种类型的响应数据。可以使用对象作为方法的返回类型，并使用Spring Boot自动将其转换为JSON响应。可以使用`ResponseEntity`类来自定义响应状态码、响应头等。

这只是在Spring Boot框架中编写REST API的基本示例，实际情况可能因具体的业务需求和设计而有所不同。建议参考Spring Boot和Spring MVC的官方文档以获取更详细的信息和示例代码。