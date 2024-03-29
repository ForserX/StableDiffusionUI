﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

namespace SD_FXUI.Debug
{
    internal class Log
    {
        private static string currentLogFile = "";

        public static void InitLogFile()
        {
            string filename = "";

            filename += "/log/";
            filename += CodeUtils.Data();
            filename += ".log";

            currentLogFile = FS.GetWorkingDir() + filename;

            Directory.CreateDirectory(FS.GetWorkingDir() + "/log/");

            File.Create(currentLogFile).Dispose();
        }


        public static string GetMessage()
        {
            return string.Empty;
        }

        public static void SendMessageToFile(string msg, bool newLine = true, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filepath = null, [CallerMemberName] string memberName = null)
        {
            if (newLine) File.AppendAllText(currentLogFile, filepath + ":" + lineNumber + ":" + memberName + " \"" + msg + "\"" + "\n");
            else File.AppendAllText(currentLogFile, filepath + ":" + lineNumber + ":" + memberName + " \"" + msg + "\"");
        }

        public static void SendMessageToFileFromHost(string msg, bool newLine = true)
        {
            if (newLine) File.AppendAllText(currentLogFile, " ~" + msg + "~" + "\n");
            else File.AppendAllText(currentLogFile, " ~" + msg + "~");
        }
    }
}
