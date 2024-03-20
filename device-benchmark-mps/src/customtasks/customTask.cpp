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

#include <customTask.h>

#include <map>

std::map<std::string, ClassInfo *> *CustomTask::classInfoMap;

void CustomTask::Register(ClassInfo *ci)
{
    if (classInfoMap == nullptr)
    {
        CustomTask::classInfoMap = new std::map<std::string, ClassInfo *>();
    }
    if (NULL != ci && classInfoMap->find(ci->m_className) == classInfoMap->end())
    {
        classInfoMap->insert(std::map<std::string, ClassInfo *>::value_type(ci->m_className, ci));
    }
}

CustomTask *CustomTask::CreateObject(std::string name, SyncType syncType)
{
    std::map<std::string, ClassInfo *>::const_iterator iter = classInfoMap->find(name);
    if (iter != classInfoMap->end())
    {
        return iter->second->CreateObject(name, syncType);
    }
    return NULL;
}

ClassInfo::ClassInfo(const std::string className, ObjectConstructorFn ctor)
    : m_className(className), m_objectConstructor(ctor)
{
    CustomTask::Register(this);
}

ClassInfo::ClassInfo() { return; }

CustomTask *ClassInfo::CreateObject(std::string name, SyncType syncType) const
{
    return m_objectConstructor ? (*m_objectConstructor)(name, syncType) : 0;
}