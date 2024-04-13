## 设置环境变量

在命令行中，你可以使用 `setx` 命令来设置环境变量。以下是在 Windows 系统中使用命令行设置环境变量的步骤：

1. 打开命令提示符（Command Prompt）或 PowerShell。
2. 使用以下命令来设置用户级别的环境变量：

   ```shell
   setx VariableName "VariableValue"
   ```

   注意，将 "VariableName" 替换为要设置的环境变量的名称，将 "VariableValue" 替换为要设置的环境变量的值。例如：

   ```shell
   setx PATH "C:\MyFolder;%PATH%"
   ```

   上述示例将将 "C:\MyFolder" 添加到 PATH 环境变量的开头，并保留原有的 PATH 值。

3. 如果要设置系统级别的环境变量，需要使用 `/m` 参数：

   ```shell
   setx /m VariableName "VariableValue"
   ```

   注意，系统级别的环境变量需要管理员权限才能设置。

4. 执行命令后，会显示一个成功的消息。

   ```
   SUCCESS: Specified value was saved.
   ```

请注意以下事项：
- 设置环境变量后，新的环境变量将在新打开的命令提示符或 PowerShell 窗口中生效。
- 对于已经打开的命令提示符或 PowerShell 窗口，需要关闭并重新打开才能使新的环境变量生效。
- 使用 `setx` 命令设置的环境变量是持久的，会在下次登录时仍然存在。

使用 `setx` 命令设置环境变量时要小心，确保输入正确的名称和值，并避免对现有的环境变量进行误操作。

### set和setx

相反，使用 `set` 命令设置的环境变量是临时的，仅在当前命令提示符或 PowerShell 会话中有效。一旦关闭会话，这些环境变量将被清除，不会在下次登录时保留。

使用 `set` 命令设置环境变量的语法如下：

```shell
set VariableName=VariableValue
```

例如：

```shell
set TEMP=C:\Temp
```

上述命令会将临时环境变量 `TEMP` 设置为 `C:\Temp`。这个环境变量仅在当前命令提示符或 PowerShell 会话中有效，关闭会话后会被清除。

请注意，使用 `set` 命令设置的环境变量仅对当前会话有效，并不会持久保存。如果希望在下次登录时仍然存在的环境变量，应该使用 `setx` 命令。

## 文件复制

在 Windows 系统上，你可以使用批处理脚本（.bat）来实现将一个文件夹下的所有文件复制到另一个文件夹的操作。你可以按照以下步骤进行操作：

1. 打开一个文本编辑器，比如记事本。
2. 在文本编辑器中，输入以下命令并保存为一个 .bat 文件，比如 `copy_files.bat`：

```bat
@echo off
set "source_folder=C:\path\to\source\folder"
set "destination_folder=C:\path\to\destination\folder"
xcopy /s /i /y "%source_folder%\*" "%destination_folder%\"
```

请确保将 `source_folder` 和 `destination_folder` 的值替换为实际的源文件夹和目标文件夹的路径。

3. 双击运行该 .bat 文件，它将会将源文件夹下的所有文件复制到目标文件夹下。

上述批处理脚本中使用了 `xcopy` 命令来执行复制操作。`/s` 参数表示复制目录和子目录，`/i` 参数表示如果目标文件夹不存在，则视为目标是一个目录，`/y` 参数表示如果命名冲突直接覆盖。`"%source_folder%\*"` 表示复制源文件夹下的所有文件和子目录（包括隐藏文件），`"%destination_folder%\"` 表示复制到目标文件夹。

确保在替换路径时使用实际的文件夹路径，并确保你有足够的权限来读取源文件夹和写入目标文件夹。

## @echo off

`@echo off` 是一个用于批处理脚本的命令。它用于在批处理脚本运行时关闭命令回显功能。

命令回显是指在运行批处理脚本时，显示每个执行的命令本身及其输出结果。通过使用 `@echo off` 命令，你可以禁止显示批处理脚本中每个命令的执行结果，使得脚本执行时只显示命令的输出结果而不显示命令本身。

具体来说，`@` 符号用于禁止显示当前行中的命令本身，而 `echo off` 则是关闭后续命令的回显。这样可以使得脚本在运行时更加整洁，只显示必要的输出信息，而不会将脚本中的每个命令都显示出来。

例如，在以下的批处理脚本中：

```bat
@echo off
echo Hello, World!
```

如果没有使用 `@echo off`，脚本运行时会显示如下输出：

```text
@echo off
Hello, World!
```

而使用了 `@echo off`，脚本运行时只会显示如下输出：

```text
Hello, World!
```

总之，`@echo off` 命令用于在批处理脚本中关闭命令的回显，使得脚本执行时只显示输出结果而不显示命令本身。