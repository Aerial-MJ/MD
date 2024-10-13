# IOC注入

## javaBean

在Java编程中，JavaBean是一种**符合特定规范的Java类**。它是一种可重用组件，通常用于在Java应用程序中封装数据和业务逻辑。JavaBean类必须遵循一组命名约定和设计模式，以便能够与其他J

ava组件进行交互，例如图形用户界面（GUI）工具和持久化框架。

以下是JavaBean的一些特征和规范：

1. 命名约定：JavaBean类的名称应以大写字母开头，使用驼峰命名法（camel case）。

2. 公共无参数构造函数：JavaBean类必须提供一个公共的无参数构造函数，以便其他组件可以实例化该类。

3. 属性访问器方法：JavaBean类应该为每个属性（字段）提供公共的getter和setter方法，以便其他组件可以读取和修改属性的值。

   - Getter方法命名约定：以 "get" 或者 "is" 开头，后面跟随属性名称，例如：`getName()` 或 `isStudent()`
   - Setter方法命名约定：以 "set" 开头，后面跟随属性名称，例如：`setName(String name)` 或 `setStudent(boolean student)`

4. 序列化支持：JavaBean类可以实现`java.io.Serializable`接口，以支持对象的序列化和反序列化。

JavaBean的设计目标是提供一种可重用和互操作的组件模型，可以在不同的Java环境中使用。它广泛应用于Java图形界面开发、持久化框架（如Hibernate）和其他领域，以简化代码编写和提高代码的可维护性。

==bean先创建再注入==

## DI 依赖注入

在Spring框架中，你可以使用XML配置文件来创建Bean。下面是使用XML配置文件创建Bean的步骤：

1. 创建XML配置文件（通常以`.xml`为扩展名），例如`applicationContext.xml`。

2. 在XML配置文件中定义Bean，使用`<bean>`元素来描述Bean的配置。以下是一个示例：

```xml
<beans xmlns="http://www.springframework.org/schema/beans"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.springframework.org/schema/beans
        http://www.springframework.org/schema/beans/spring-beans.xsd">

    <bean id="myBean" class="com.example.MyBean">
        <!-- 设置Bean的属性 -->
        <property name="propertyName" value="propertyValue" />
    </bean>

</beans>
```

在上面的示例中，通过`<bean>`元素定义了一个名为`myBean`的Bean，它的类为`com.example.MyBean`。可以使用`<property>`元素设置Bean的属性，其中`name`属性指定属性名，`value`属性指定属性值。

3. 在应用程序的代码中，使用Spring的`ApplicationContext`来加载XML配置文件并获取Bean的实例。例如：

```java
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class MyApp {
    public static void main(String[] args) {
        // 加载XML配置文件
        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");

        // 获取Bean实例
        MyBean myBean = (MyBean) context.getBean("myBean");

        // 使用Bean
        myBean.doSomething();
    }
}
```

在上面的示例中，通过`ClassPathXmlApplicationContext`类加载了`applicationContext.xml`配置文件，并使用`getBean()`方法获取了`myBean`的实例。然后可以通过该实例调用Bean的方法。

这只是一个简单的示例，实际上XML配置文件还支持更多的配置选项和功能，例如构造函数注入、依赖注入等。你可以根据具体需求和Spring的文档进一步学习和了解XML配置文件的使用。

### setter方式注入

使用Spring中的属性注入（Property Injection）。在这种注入类型中，通过使用`<property>`元素在XML配置文件中设置Bean的属性。

具体来说，`<property>`元素使用`name`属性指定要设置的属性名，使用`value`属性指定属性的值。在上面的示例中，通过`<property>`元素将名为`propertyName`的属性设置为`propertyValue`。

以下是一个更完整的示例：

```xml
<bean id="myBean" class="com.example.MyBean">
    <!-- 设置Bean的属性 -->
    <property name="propertyName" value="propertyValue" />
    <property name="anotherProperty" ref="anotherBean" />
</bean>

<bean id="anotherBean" class="com.example.AnotherBean" />
```

在上面的示例中，通过两个`<property>`元素设置了两个属性。第一个`<property>`元素将名为`propertyName`的属性设置为`propertyValue`，第二个`<property>`元素将名为`anotherProperty`的属性设置为引用了名为`anotherBean`的另一个Bean。

在相应的`MyBean`类中，需要提供与属性名对应的setter方法来接受注入的值。例如：

```java
public class MyBean {
    private String propertyName;
    private AnotherBean anotherProperty;

    public void setPropertyName(String propertyName) {
        this.propertyName = propertyName;
    }

    public void setAnotherProperty(AnotherBean anotherProperty) {
        this.anotherProperty = anotherProperty;
    }

    // ...
}
```

在上面的示例中，我们提供了`setPropertyName()`和`setAnotherProperty()`方法来接受注入的属性值。

通过属性注入，你可以在XML配置文件中灵活地设置Bean的属性，并将值直接注入到对应的setter方法中。这种方式常用于注入基本类型的属性值和引用其他Bean的属性值。

### 使用构造函数注入

如果想手动确定参数的位置，你可以在XML配置文件中使用`<constructor-arg>`元素的`index`属性来指定参数的位置。以下是一个示例：

```xml
<bean id="myBean" class="com.example.MyBean">
    <constructor-arg index="1" value="value2" />
    <constructor-arg index="0" value="value1" />
</bean>
```

在上面的示例中，假设`MyBean`类有一个构造函数，接受两个字符串参数。使用两个`<constructor-arg>`元素，并分别通过`index`属性指定了参数的位置。

下面是相应的`MyBean`类的定义：

```java
public class MyBean {
    private String property1;
    private String property2;

    public MyBean(String property1, String property2) {
        this.property1 = property1;
        this.property2 = property2;
    }

    public void doSomething() {
        System.out.println("Property 1 value: " + property1);
        System.out.println("Property 2 value: " + property2);
    }
}
```

在上面的示例中，`MyBean`类具有一个构造函数，接受两个字符串参数。在`doSomething()`方法中，我们打印出这两个属性的值。

通过上述配置和`MyBean`类的定义，当Spring容器启动时，会实例化`MyBean`对象并将构造函数参数按照指定的位置传递给它。然后，你可以从Spring容器中获取`MyBean`的实例并调用其方法，如下所示：

```java
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class MyApp {
    public static void main(String[] args) {
        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");

        MyBean myBean = (MyBean) context.getBean("myBean");

        myBean.doSomething();
    }
}
```

在上面的示例中，我们从Spring容器中获取了`myBean`的实例，并调用其`doSomething()`方法。在输出中，你将看到两个属性的值按照指定的位置被正确设置和显示。

通过使用`<constructor-arg>`元素的`index`属性，你可以手动确定参数的位置，以满足特定的需求。

### 使用注解注入

**@Component：可以用于注册所有bean**

**@Repository：主要用于注册pojo层的bean**

**@Controller：主要用于注册控制层的bean**

**@Service：主要用于注册服务层的bean**

**@Mapper：主要用于注册Dao层的bean**

通常情况下我们在创建spring项目的时候在xml配置文件中都会配置这个标签，配置完这个标签后，spring就会去自动扫描base-package对应的路径或者该路径的子包下面的java文件，如果扫描到文件中带有@Service,@Component,@Repository,@Controller等这些注解的类，则把这些类注册为bean
注：在注解后加上例如@Component(value=”abc”)时，注册的这个类的bean的id就是adc.

使用注解方式： 在Bean类中使用Spring的注解来标注属性，从而实现自动注入。常用的注解有`@Autowired`、`@Value`等。以下是一个示例：

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

public class MyBean {
    @Autowired
    private OtherBean otherBean;

    @Value("propertyValue")
    private String propertyName;

    // ...
}
```

在上面的示例中，通过`@Autowired`注解实现了对`otherBean`属性的自动注入，通过`@Value`注解指定了`propertyName`属性的值。

**通过byTYPE（@Autowired）、byNAME（@Resource）的方式获取Bean**

#### @Autowired

##### @Autowired(required = false)

在Java中，`@Autowired(required = false)` 是一个用于依赖注入的注解，用于告诉容器在注入依赖时，如果找不到匹配的依赖项，不要抛出异常。如果将 `required` 设置为 `false`，容器会尝试自动注入依赖，如果找不到匹配的依赖项，则将依赖项设置为 `null` 或者忽略该依赖项。

通常情况下，如果没有找到匹配的依赖项，`@Autowired` 注解会导致应用程序启动失败并抛出异常。但是通过将 `required` 设置为 `false`，可以避免这种情况，并且允许应用程序继续运行，即使依赖项没有被正确注入。

下面是一个示例：

```java
@Autowired(required = false)
private SomeDependency someDependency;
```

在上面的代码中，如果容器找不到与 `SomeDependency` 类型匹配的依赖项，将不会抛出异常，而是将 `someDependency` 设置为 `null`。

需要注意的是，如果将 `required` 设置为 `false`，应该在代码中检查依赖项是否为 `null`，以避免空指针异常。

##### @Autowired和@Qualified

当使用`@Autowierd`和`@Qualified`一起时，通常是在Spring框架中进行依赖注入时使用的注解。`@Autowired`用于自动装配（即自动注入）依赖项，而`@Qualified`用于指定要注入的特定bean。

下面是一个使用`@Autowired`和`@Qualified`一起的示例：

```java
public class ExampleService {

   @Autowired
   @Qualified("specificBean")
   private SomeBean someBean;

   // 其他代码
}
```

在上面的示例中，`ExampleService`类中的`someBean`字段使用了`@Autowired`和`@Qualified`注解。`@Autowired`告诉Spring要自动装配这个字段，而`@Qualified("specificBean")`告诉Spring要注入一个名为"specificBean"的特定bean。

假设在Spring的应用上下文（application context）中**有多个`SomeBean`类型的bean**，而你想要注入名为"specificBean"的那个bean，那么你可以使用`@Qualified`注解来明确指定要注入的bean。

请注意，`@Qualified`注解的参数是一个字符串，用于指定特定bean的名称或限定符（qualifier）。这个名称或限定符需要与你想要注入的bean的名称或限定符相匹配。

这只是一个简单的示例，实际上`@Autowired`和`@Qualified`可以在更复杂的依赖注入场景中使用，以满足更精确的注入需求。

**注解创建相同类型的bean**

要创建具有多个相同类型的`SomeBean`的bean，你可以使用Spring框架中的`@Qualifier`注解来为这些bean提供限定符（qualifier）。通过使用`@Qualifier`注解，你可以为每个bean指定不同的限定符，并在注入时使用相应的限定符来区分它们。

**下面是一个创建具有多个`SomeBean`类型的bean并使用`@Qualifier`的**

**示例1**

```java
@Component
@Qualifier("bean1")
public class SomeBeanImpl1 implements SomeBean {
    // 实现代码
}

@Component
@Qualifier("bean2")
public class SomeBeanImpl2 implements SomeBean {
    // 实现代码
}
```

在上面的示例中，我们创建了两个实现了`SomeBean`接口的bean，分别是`SomeBeanImpl1`和`SomeBeanImpl2`。每个bean上都使用了`@Qualifier`注解，并为它们分别指定了不同的限定符（"bean1"和"bean2"）。

接下来，在需要注入`SomeBean`类型的地方，你可以使用`@Qualifier`注解来指定要注入的具体bean。例如：

```java
@Component
public class ExampleService {

   @Autowired
   @Qualified("bean1")
   private SomeBean someBean;

   // 其他代码
}
```

在上面的示例中，`ExampleService`类中的`someBean`字段使用了`@Autowired`和`@Qualifier("bean1")`注解。这表示要注入具有限定符为"bean1"的`SomeBean`类型的bean。

**示例2**

通过使用不同的`@Qualifier`限定符，你可以明确指定要注入的具体bean，以满足具体的需求。这样，当有多个相同类型的bean存在时，Spring框架就能根据限定符来区分它们并进行正确的注入。

```java
//在UserDaoImpl类中
@Component("userDao1")
public class UserDaoImpl implements UserDao {
    public void addUser() {
        System.out.println("添加用户");
    }
}

//在UserDaoImpl2类中
@Component("userDao2")
public class UserDaoImpl2 implements UserDao {
    public void addUser() {
        System.out.println("添加新的学生");
    }
}

//这样配置没指定bean名称是会出错的
	@Autowired
    private UserDao userDao;
//所以要@Qualifier指定bean名称
	@Autowired
    @Qualifier("userDao1")
    private UserDao userDao;

//运行结果
```

#### @Resource

当使用 `@Resource` 注解时，可以通过 `name` 参数指定要注入的 Bean 的名称。下面是一个示例：

```java
public class MyClass {
    @Resource(name = "myBean")
    private MyBean myBean;
    
    public void doSomething() {
        myBean.doSomething();
    }
}

@Component("myBean")
public class MyBean {
    public void doSomething() {
        System.out.println("Doing something...");
    }
}
```

在上面的示例中，`MyClass` 类使用 `@Resource` 注解注入了一个名为 "myBean" 的 Bean，并将其赋值给 `myBean` 字段。通过在 `MyClass` 类中调用 `myBean.doSomething()` 方法，可以执行被注入 Bean 的相应操作。

请注意，使用 `@Resource` 注解时，可以通过 `name` 参数指定要注入的 Bean 的名称。在示例中，`@Resource(name = "myBean")` 表示要注入名为 "myBean" 的 Bean。

需要确保在使用 `@Resource` 注解进行注入之前，相关的 Bean 已经被创建并注册到 Spring 容器中。

### 使用@Configuration注入

使用Java配置类： 除了XML配置文件外，你还可以使用Java配置类来定义Bean和设置属性。通过在Java配置类中使用`@Configuration`、`@Bean`等注解，可以创建和配置Bean。以下是一个示例：

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {
    @Bean
    public MyBean myBean() {
        MyBean bean = new MyBean();
        bean.setProperty("propertyValue");
        return bean;
    }
}
```

在上面的示例中，通过`@Bean`注解定义了`myBean`方法，该方法返回一个`MyBean`实例，并在方法中设置了属性值。

这些方法提供了不同的方式来设置Bean的属性，你可以根据具体情况选择最适合的方法。通常，推荐使用注解方式或Java配置类，因为它们更加灵活和类型安全，且能够将配置与代码更紧密地结合在一起。

## 多例模式

当需要创建多个实例的bean时，可以使用Spring框架中的作用域（Scope）来实现多例模式。通过将作用域设置为"prototype"，每次从容器中获取该bean时都会创建一个新的实例。这样可以确保每次获取该bean时都得到一个独立的实例。

下面是一个使用多例模式的示例：

```java
@Component
@Scope("prototype")
public class SomeBean {
    // 实现代码
}
```

在上面的示例中，我们将`SomeBean`类上的作用域注解设置为`@Scope("prototype")`，这表示该bean的作用域是多例模式。每次从Spring容器中获取`SomeBean`时，都会创建一个新的实例。

接下来，可以在其他组件中注入`SomeBean`，并观察每次获取的实例是否不同：

```java
@Component
public class ExampleService {

   @Autowired
   private SomeBean someBean;

   public void doSomething() {
       // 使用someBean进行操作
   }
}
```

在上面的示例中，`ExampleService`类中的`someBean`字段使用了`@Autowired`注解来进行注入。由于`SomeBean`的作用域设置为多例模式，每次调用`ExampleService`的`doSomething()`方法时，都会获得一个新的`SomeBean`实例。

这样，无论调用多少次`doSomething()`方法，都会使用不同的`SomeBean`实例，每个实例都是独立的，彼此之间没有共享状态。

通过使用多例模式，可以满足某些场景下对多个独立实例的需求，例如线程安全性、状态隔离等。