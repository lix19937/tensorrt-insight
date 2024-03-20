/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once
#include <Task.h>

#include <map>
// TODO: set it to real flag
#define DECLARE_CLASS()                                                                                                \
  protected:                                                                                                           \
    static ClassInfo ms_classinfo;                                                                                     \
                                                                                                                       \
  public:                                                                                                              \
    static CustomTask *CreateObject(std::string name, SyncType syncType);

#define IMPLEMENT_CLASS(interface_name, class_name)                                                                    \
    ClassInfo   class_name::ms_classinfo(interface_name, (ObjectConstructorFn)class_name::CreateObject);               \
    CustomTask *class_name::CreateObject(std::string name, SyncType syncType)                                          \
    {                                                                                                                  \
        return new class_name(name, syncType);                                                                         \
    };

class ClassInfo;
class CustomTask;
typedef CustomTask *(*ObjectConstructorFn)(std::string, SyncType);

class ClassInfo
{
  public:
    ClassInfo(const std::string className, ObjectConstructorFn ctor);
    ClassInfo();
    CustomTask *CreateObject(std::string name, SyncType syncType) const;

  public:
    std::string         m_className;
    ObjectConstructorFn m_objectConstructor;
};

class CustomTask : public Task
{
  public:
    // CustomTask() = delete;
    virtual ~CustomTask() {}
    static void                                Register(ClassInfo *ci);
    static CustomTask *                        CreateObject(std::string name, SyncType syncType);
    static std::map<std::string, ClassInfo *> *classInfoMap;
};
