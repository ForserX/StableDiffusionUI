// XUI.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <Windows.h>
#include <filesystem>
#include <string>

std::string GetTryFileDirectory()
{
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");

	return std::string(buffer).substr(0, pos);
}

int main()
{
	std::string Dir = std::move(GetTryFileDirectory());

	std::filesystem::current_path(Dir); //setting path
	system("@start bins\\SD-FXUI.exe");

	return 0;
}