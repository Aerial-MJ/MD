# SpringMVC

## 处理流程

当客户端发送一个HTTP请求到Spring MVC应用程序时，Spring MVC会按照以下步骤处理请求：

1. 客户端发送请求：客户端（例如浏览器）发送HTTP请求到Spring MVC应用程序的前端控制器，通常是通过URL访问一个特定的资源。

2. 前端控制器接收请求：前端控制器（DispatcherServlet）是Spring MVC框架的中央调度器，它接收所有的请求并将其分派给相应的处理器进行处理。

3. 处理器映射器（Handler Mapping）：前端控制器将请求交给处理器映射器，处理器映射器根据配置的映射规则，确定请求应该由哪个处理器（Controller）来处理。

4. 处理器适配器（Handler Adapter）：处理器适配器负责将请求发送给处理器，并将处理器的执行结果返回给前端控制器。Spring MVC支持多种类型的处理器，例如注解驱动的控制器、基于接口的控制器等。

5. 控制器处理请求：处理器（Controller）是一个Java类，它包含了处理请求的业务逻辑。处理器根据请求的参数和路径执行相应的操作，并生成一个模型（Model）对象，存储处理结果。

6. 模型数据准备：处理器可以从数据库、外部服务等数据源获取数据，并将数据存储在模型对象中，以供视图（View）使用。

7. 视图解析器（View Resolver）：前端控制器将模型对象发送给视图解析器，视图解析器根据配置的规则，确定应该使用哪个视图来呈现模型数据。

8. 视图渲染：视图负责将模型数据呈现给客户端。它可以是一个JSP页面、Thymeleaf模板、Freemarker模板等，根据模型数据生成HTML响应。

9. 响应返回给客户端：视图生成HTML响应后，前端控制器将其返回给客户端，完成请求-响应周期。

在整个处理流程中，还可以使用拦截器（Interceptor）、数据绑定、数据验证等特性进行额外的处理和操作，以满足具体的业务需求。

总结来说，Spring MVC的处理流程是：前端控制器接收请求，根据处理器映射器确定请求由哪个处理器处理，处理器执行业务逻辑并生成模型数据，视图解析器确定使用哪个视图进行呈现，最后将视图响应返回给客户端。这个流程允许开发人员将应用程序的不同层（控制器、服务、模型和视图）进行解耦，并提供了灵活的扩展和定制能力。

![img](../../../Image/图片1.png)

## 实例

**ProductController**

首先，我们创建一个控制器类 `ProductController`，它处理商品列表页面的请求：

```java
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class ProductController {

    @RequestMapping("/products")
    public String productList(Model model) {
        // 从数据库或其他数据源获取商品列表数据
        List<Product> products = productService.getProducts();
        
        // 将商品列表数据存储在模型中
        model.addAttribute("products", products);
        
        // 返回视图名称
        return "productList";
    }
}
```

在上述代码中，`ProductController` 类使用 `@Controller` 注解标识为一个控制器，`@RequestMapping("/products")` 注解指定了处理的URL路径为 "/products"。`productList()` 方法处理该请求，获取商品列表数据，并将数据存储在模型中。最后，该方法返回视图名称 "productList"。

**WebMvcConfigurer**

接下来，我们需要配置 Spring MVC 的相关组件和映射规则。可以使用 XML 配置或者 Java 配置方式，以下是一个简单的 Java 配置示例：

```java
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.servlet.config.annotation.ViewResolverRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
@EnableWebMvc
public class MvcConfig implements WebMvcConfigurer {

    @Override
    public void configureViewResolvers(ViewResolverRegistry registry) {
        registry.jsp("/WEB-INF/views/", ".jsp");
        //视图解析器根据配置的规则，确定应该使用哪个视图来呈现模型数据。
    }
}
```

上述代码中，`MvcConfig` 类使用 `@Configuration` 注解表示这是一个配置类，`@EnableWebMvc` 注解启用了Spring MVC。`configureViewResolvers()` 方法配置了视图解析器，指定了视图文件的前缀和后缀。

**DispatcherServlet**

最后，在项目的web.xml文件中配置前端控制器 `DispatcherServlet`：

```xml
<web-app>
    <servlet>
        <servlet-name>dispatcherServlet</servlet-name>
        <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
        <init-param>
            <param-name>contextConfigLocation</param-name>
            <param-value>/WEB-INF/applicationContext.xml</param-value>
        </init-param>
        <load-on-startup>1</load-on-startup>
    </servlet>

    <servlet-mapping>
        <servlet-name>dispatcherServlet</servlet-name>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
```

上述代码中，我们将 `DispatcherServlet` 配置为处理所有请求，并指定了 `contextConfigLocation` 参数来指定 Spring MVC 的配置文件位置。

**productList.jsp**

首先，在项目的 `WEB-INF/views` 目录下创建名为 `productList.jsp` 的JSP文件，然后在该文件中编写以下代码：

```jsp
<%@ page contentType="text/html; charset=UTF-8" %>
<html>
<head>
    <title>Product List</title>
</head>
<body>
    <h1>Product List</h1>
    <table>
        <tr>
            <th>Product Name</th>
            <th>Price</th>
        </tr>
        <c:forEach items="${products}" var="product">
            <tr>
                <td>${product.name}</td>
                <td>${product.price}</td>
            </tr>
        </c:forEach>
    </table>
</body>
</html>
```

在上述代码中，我们使用 JSP 标签 `<c:forEach>` 遍历 `${products}` 集合，该集合存储了从控制器传递过来的商品列表数据。`${product.name}` 和 `${product.price}` 分别显示了商品的名称和价格。

请确保在 JSP 文件开头添加 `<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>` 来导入 JSTL 的标签库。

这样，当控制器处理 `/products` 请求时，会将商品列表数据存储在模型中，并返回视图名称 `"productList"`。前端控制器将视图名称解析为实际的 JSP 文件路径，并将模型数据传递给 JSP 文件。最终，浏览器将显示商品列表页面，并按照模型中的数据进行呈现。

## thymeleaf

如果想在Thymeleaf中编写类似的代码来展示商品列表，可以按照以下方式进行：**(不使用jsp，本质上jsp和thymeleaf作用是一样的)**

1. 首先，在项目中添加Thymeleaf的依赖，例如通过Maven：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-thymeleaf</artifactId>
</dependency>
```

2. 在Spring配置中启用Thymeleaf，例如使用Java配置方式：

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.ViewResolver;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.web.servlet.view.InternalResourceViewResolver;
import org.thymeleaf.spring5.SpringTemplateEngine;
import org.thymeleaf.spring5.view.ThymeleafViewResolver;
import org.thymeleaf.templatemode.TemplateMode;
import org.thymeleaf.templateresolver.ServletContextTemplateResolver;

@Configuration
@EnableWebMvc
public class ThymeleafConfig implements WebMvcConfigurer {

    @Bean
    public ViewResolver viewResolver() {
        ThymeleafViewResolver resolver = new ThymeleafViewResolver();
        resolver.setTemplateEngine(templateEngine());
        return resolver;
    }

    @Bean
    public SpringTemplateEngine templateEngine() {
        SpringTemplateEngine engine = new SpringTemplateEngine();
        engine.setTemplateResolver(templateResolver());
        return engine;
    }

    @Bean
    public ServletContextTemplateResolver templateResolver() {
        ServletContextTemplateResolver resolver = new ServletContextTemplateResolver();
        resolver.setPrefix("/WEB-INF/views/");
        resolver.setSuffix(".html");
        resolver.setTemplateMode(TemplateMode.HTML);
        return resolver;
    }
}
```

在上述代码中，我们配置了Thymeleaf的视图解析器，并设置了模板引擎和模板解析器。视图解析器将解析Thymeleaf模板并呈现给客户端。

3. 创建名为 `productList.html` 的Thymeleaf模板文件，放置在 `/WEB-INF/views/` 目录下，然后在该文件中编写以下代码：

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Product List</title>
</head>
<body>
    <h1>Product List</h1>
    <table>
        <tr>
            <th>Product Name</th>
            <th>Price</th>
        </tr>
        <tr th:each="product : ${products}">
            <td th:text="${product.name}"></td>
            <td th:text="${product.price}"></td>
        </tr>
    </table>
</body>
</html>
```

在Thymeleaf模板中，我们使用 `th:` 命名空间来使用Thymeleaf的属性和表达式。`th:each` 属性用于迭代 `${products}` 集合，`product` 是迭代的当前对象。`${product.name}` 和 `${product.price}` 表达式用于获取商品对象的名称和价格。

4. 在控制器类中，将模型数据添加到响应的属性中，例如：

```java
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class ProductController {

    @RequestMapping("/products")
    public String productList(Model model) {
        // 从数据库或其他数据源获取商品列表数据
        List<Product> products = productService.getProducts();
        
        // 将商品列表数据存储在模型中
        model.addAttribute("products", products);
        
        // 返回视图名称
        return "productList";
    }
}
```

在上述代码中，我们使用 `model.addAttribute()` 方法将商品列表数据存储在名为 "products" 的属性中，该属性将在Thymeleaf模板中使用。
