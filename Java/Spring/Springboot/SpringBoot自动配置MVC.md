# SpringBoot自动配置MVC

SpringBoot在自动配置很多组件的时候，先看容器中有没有用户自己配置的（如果用户自己配置@bean），如果有就用用户配置的，如果没有就用自动配置的。 如果有些组件可以存在多个，比如我们的视图解析器，就将用户配置的和自己默认的组合起来。

我们要扩展SpringMVC，官方就推荐我们这么去使用，既保SpringBoot留所有的自动配置，也能用我们扩展的配置！

## MVC静态资源导入

```java
@Override
public void addResourceHandlers(ResourceHandlerRegistry registry) {
   if (!this.resourceProperties.isAddMappings()) {
      logger.debug("Default resource handling disabled");
      return;
   }
   addResourceHandler(registry, "/webjars/**", "classpath:/META-INF/resources/webjars/");
   addResourceHandler(registry, this.mvcProperties.getStaticPathPattern(), (registration) -> {
      registration.addResourceLocations(this.resourceProperties.getStaticLocations());
      if (this.servletContext != null) {
         ServletContextResource resource = new ServletContextResource(this.servletContext, SERVLET_LOCATION);
         registration.addResourceLocations(resource);
      }
   });
}
```
**还可以配置例如视图解析器等MVC的内容**

## 全面接管SpringMVC
**如果你想自己写SpringMVC配置，可以采用全面接管的方式**

官方文档：

If you want to take complete control of Spring MVC you can add your own @Configuration annotated with @EnableWebMvc.
全面接管即：SpringBoot对SpringMVC的自动配置不需要了，所有都是我们自己去配置！

**只需在我们的配置类中要加一个@EnableWebMvc。**

这里发现它是导入了一个类，我们可以继续进去看
```java
@Import({DelegatingWebMvcConfiguration.class})
public @interface EnableWebMvc {
}
```
2、它继承了一个父类 WebMvcConfigurationSupport

```java
public class DelegatingWebMvcConfiguration extends WebMvcConfigurationSupport {
  // ......
}
```
3、我们来回顾一下Webmvc自动配置类
```java
@Configuration(proxyBeanMethods = false)
@ConditionalOnWebApplication(type = Type.SERVLET)
@ConditionalOnClass({ Servlet.class, DispatcherServlet.class, WebMvcConfigurer.class })
// 这个注解的意思就是：容器中没有这个组件的时候，这个自动配置类才生效
@ConditionalOnMissingBean(WebMvcConfigurationSupport.class)
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE + 10)
@AutoConfigureAfter({ DispatcherServletAutoConfiguration.class, TaskExecutionAutoConfiguration.class,
    ValidationAutoConfiguration.class })
public class WebMvcAutoConfiguration {
    
}
```
**总结一句话：@EnableWebMvc将WebMvcConfigurationSupport组件导入进来了,导致springboot的WebMvcAutoConfiguration无法发挥作用，只能使用自己创建的SpringMVC的控制类。**

