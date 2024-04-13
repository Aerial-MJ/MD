# Spring创建

## 使用maven

Maven（Apache Maven）是一个开源的项目管理和构建工具，用于构建、发布和管理Java项目。它提供了一种**结构化的方式来定义项目的构建过程，管理项目依赖关系，并协调项目的构建、测试和部署等任务。**

下面是 Maven 的一些主要特点和功能：

1. **项目对象模型（Project Object Model，POM）**：Maven 使用 POM 文件来描述项目的结构、依赖关系和构建配置。POM 是一个 XML 文件，包含了项目的元数据、构建配置、依赖项列表等信息。
2. **依赖管理**：Maven 提供了依赖管理机制，可以通过在 POM 文件中声明依赖项来管理项目所需的外部库或模块。Maven 会自动下载所需的依赖，并构建正确的依赖关系图。
3. **标准化的项目布局**：Maven 定义了一套标准的项目布局，可以帮助开发者组织和管理项目的源代码、资源文件和其他构建产物。这样，开发者可以更容易地理解项目的结构，并快速找到所需的文件。
4. **生命周期和插件机制**：Maven 定义了一套生命周期和插件机制，用于执行项目的各个构建阶段和任务。开发者可以通过配置插件来自定义构建过程，执行代码编译、测试、打包、部署等任务。
5. **远程仓库和版本管理**：Maven 使用远程仓库来存储和获取项目的依赖项和插件。它支持多种远程仓库管理工具，如 Maven Central Repository，可以轻松地共享和获取各种开源库。

使用 Maven 可以带来许多好处，例如简化项目的构建和管理、提高开发效率、统一构建流程、自动化依赖管理和版本控制等。通过 Maven，开发者可以更专注于项目的开发，而无需手动处理繁琐的构建和依赖管理任务。

需要注意的是，Maven 是基于约定优于配置的原则，它有一套默认的约定和配置，遵循一致的项目结构和命名规则。这使得不同的 Maven 项目具有一致性，并且可以更容易地理解和维护。同时，Maven 也允许开发者根据自己的需求进行配置和定制。

### 配置步骤
**在pom.xml中进行如下配置**
```xml
 <dependencies>
        <!-- https://mvnrepository.com/artifact/javax.servlet/javax.servlet-api -->
        <!-- https://mvnrepository.com/artifact/org.springframework/spring-beans -->
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-beans</artifactId>
            <version>5.3.10</version>
        </dependency>
 
        <!-- https://mvnrepository.com/artifact/org.springframework/spring-core -->
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-core</artifactId>
            <version>5.3.10</version>
        </dependency>
 
        <!-- https://mvnrepository.com/artifact/org.springframework/spring-context -->
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-context</artifactId>
            <version>5.3.10</version>
        </dependency>
 
        <!-- https://mvnrepository.com/artifact/org.springframework/spring-expression -->
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-expression</artifactId>
            <version>5.3.10</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13</version>
            <scope>compile</scope>
        </dependency>
 
    </dependencies>
```
**编写Spring实体类和测试类**
```java
//Spring5Test类
 
import org.junit.Test;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
 
public class Spring5Test {
    @Test
    public  void test(){
        //加载spring配置文件
        ApplicationContext context =new ClassPathXmlApplicationContext("bean1.xml");
 
        //获取配置创建的对象
        User user=context.getBean("user",User.class);
 
        user.add();
    }
 
}
 
//User类
public class User {
    public void add(){
        System.out.println("hello Spring");
    }
 
}
```

**在resoures文件夹下配置xml文件**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
 
    <bean id="user" class="SpingDemo.User"></bean>
 
</beans>
```

## 不使用maven

**spring的创建可以不依赖于maven**

可以使用其他构建工具或手动配置来创建和管理 Spring 项目，而不依赖于 Maven。尽管 Maven 是一个广泛使用的构建工具，并且与 Spring 集成得很好，但并不是使用 Spring 的唯一选择。

以下是一些替代 Maven 的构建工具和方法：

1. **Gradle**：Gradle 是另一个流行的构建工具，它与 Maven 相比提供了更灵活和强大的构建配置。你可以使用 Gradle 构建和管理 Spring 项目，通过 Gradle 脚本定义项目的依赖关系、任务和构建过程。
2. **Ant**：Ant 是一个基于 XML 的构建工具，也可以用于构建 Spring 项目。你可以编写 Ant 脚本来定义项目的编译、打包和部署等构建任务。
3. **手动配置**：如果你不想使用构建工具，也可以手动配置和管理项目的构建过程。这种方法需要你手动编译源代码、处理依赖关系和打包项目。你可以使用命令行工具或集成开发环境（IDE）来辅助进行这些操作。

无论你选择使用哪种构建方法，重要的是确保项目的依赖关系正确管理、构建过程可重复，并且能够生成可执行的应用程序或部署包。Spring 提供了灵活的配置选项，可以与各种构建工具集成或手动配置，以满足你的项目需求。

## spring和springboot在pom.xml上的区别

Spring 和 Spring Boot 是两个相关的框架，它们有以下区别：

1. **复杂性和简化程度**：Spring 是一个全功能的企业级应用程序开发框架，提供了广泛的功能和灵活性，但需要进行大量的配置和设置。而 Spring Boot 是基于 Spring 的开发框架，旨在简化 Spring 应用程序的初始化和配置过程，提供了自动配置和约定大于配置的原则，使得开发者能够更快速地搭建和运行应用程序。

2. **依赖管理**：在 Spring 中，你需要手动管理项目的依赖项，通过 Maven 或 Gradle 配置项目的依赖关系。而 Spring Boot 使用了约定大于配置的方式，**它提供了一组预定义的 Starter 依赖，你只需在项目的配置文件中声明所需的 Starter 依赖，Spring Boot 将自动管理依赖的版本和传递依赖。**

3. **配置方式**：在 Spring 中，你需要手动配置许多组件、Bean 和其他功能，通常使用 XML 配置文件或 Java 注解来定义和配置。而 Spring Boot 使用了自动配置的机制，根据项目的依赖和环境，自动配置和组装所需的组件和功能。你可以通过修改少量的配置属性来覆盖默认行为，从而定制应用程序的行为。

4. **开发体验**：由于 Spring Boot 提供了自动配置和约定大于配置的原则，它大大简化了应用程序的初始化和开发过程。你可以更专注于业务逻辑的实现，而无需过多关注底层的配置和管理。Spring Boot 还提供了内置的开发者工具，支持热部署、自动重启等功能，提升了开发体验和生产力。

总的来说，Spring 是一个全功能的企业级开发框架，而 Spring Boot 是建立在 Spring 基础上的用于简化 Spring 应用程序开发和配置的框架。Spring Boot 提供了自动配置、约定大于配置和依赖管理等功能，使得开发者能够更快速、更轻松地创建和运行 Spring 应用程序。

**pring Boot 是在 Spring 的基础上构建的快速开发框架，通过约定优于配置的原则和自动配置的机制，简化了应用程序的开发和部署过程。Spring Boot 可以看作是 Spring 的一种扩展和简化，使得开发者可以更快地搭建和启动 Spring 应用程序。**
