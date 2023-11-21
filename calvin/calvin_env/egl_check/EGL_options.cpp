// portions of this file are copied from GLFW egl_context.c/egl_context.h

//========================================================================
// GLFW 3.3 EGL - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2002-2006 Marcus Geelnard
// Copyright (c) 2006-2016 Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "glad/egl.h"
#include "glad/gl.h"
#include "glad/glx.h"

#include "EGL_options.h"


struct EGLInternalData2
{
	bool m_isInitialized;

	int m_windowWidth;
	int m_windowHeight;
	int m_renderDevice;


	EGLBoolean success;
	EGLint num_configs;
	EGLConfig egl_config;
	EGLSurface egl_surface;
	EGLContext egl_context;
	EGLDisplay egl_display;

	EGLInternalData2()
	    : m_isInitialized(false),
	      m_windowWidth(0),
	      m_windowHeight(0) {}
};

EGLOpenGLWindow::EGLOpenGLWindow() { m_data = new EGLInternalData2(); }

EGLOpenGLWindow::~EGLOpenGLWindow() { delete m_data; }

void EGLOpenGLWindow::createWindow(const b3gWindowConstructionInfo& ci)
{
	m_data->m_windowWidth = ci.m_width;
	m_data->m_windowHeight = ci.m_height;

	m_data->m_renderDevice = ci.m_renderDevice;

	EGLint egl_config_attribs[] = {EGL_RED_SIZE,
	                               8,
	                               EGL_GREEN_SIZE,
	                               8,
	                               EGL_BLUE_SIZE,
	                               8,
	                               EGL_DEPTH_SIZE,
	                               8,
	                               EGL_SURFACE_TYPE,
	                               EGL_PBUFFER_BIT,
	                               EGL_RENDERABLE_TYPE,
	                               EGL_OPENGL_BIT,
	                               EGL_NONE};

	EGLint egl_pbuffer_attribs[] = {
	    EGL_WIDTH,
	    m_data->m_windowWidth,
	    EGL_HEIGHT,
	    m_data->m_windowHeight,
	    EGL_NONE,
	};

	// Load EGL functions
	int egl_version = gladLoaderLoadEGL(NULL);
	if (!egl_version)
	{
		fprintf(stderr, "failed to EGL with glad.\n");
		exit(EXIT_FAILURE);
	};

	// Query EGL Devices
	const int max_devices = 32;
	EGLDeviceEXT egl_devices[max_devices];
	EGLint num_devices = 0;
	EGLint egl_error = eglGetError();
	if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) ||
	    egl_error != EGL_SUCCESS)
	{
		printf("eglQueryDevicesEXT Failed.\n");
		m_data->egl_display = EGL_NO_DISPLAY;
	} else
	{
		// default case, should always happen (for future compatibility)
		if (m_data->m_renderDevice == -1)
		{
			// check env variable
			const char* env_p = std::getenv("EGL_VISIBLE_DEVICE");

			// variable is set
			if(env_p != NULL)
			{
				m_data->m_renderDevice = std::atoi(env_p);
				fprintf(stderr, "EGL device choice: %d of %d (from EGL_VISIBLE_DEVICE)\n", m_data->m_renderDevice, num_devices);

            } else {
                fprintf(stderr, "EGL device choice: %d of %d.\n", m_data->m_renderDevice, num_devices);
            } // else leave with -1

		} else
		{
			fprintf(stderr, "EGL device choice: %d of %d.\n", m_data->m_renderDevice, num_devices);
		}
	}

	// Query EGL Screens
	if (m_data->m_renderDevice == -1)
	{
		// Chose default screen, by trying all
		for (EGLint i = 0; i < num_devices; ++i)
		{
			// Set display
			EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
			                                              egl_devices[i], NULL);
			if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY)
			{
				int major, minor;
				EGLBoolean initialized = eglInitialize(display, &major, &minor);
				if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE)
				{
					m_data->egl_display = display;
					break;
				}
			}
			else
			{
				fprintf(stderr, "GetDisplay %d failed with error: %x\n", i, eglGetError());
			}
		}
	}
	else
	{
		// Chose specific screen, by using m_renderDevice
		if (m_data->m_renderDevice < 0 || m_data->m_renderDevice >= num_devices)
		{
			fprintf(stderr, "Invalid render_device choice: %d < %d.\n", m_data->m_renderDevice, num_devices);
			exit(EXIT_FAILURE);
		}
		// Set display
		EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
		                                              egl_devices[m_data->m_renderDevice], NULL);
		if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY)
		{
			int major, minor;
			EGLBoolean initialized = eglInitialize(display, &major, &minor);
			if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE)
			{
				m_data->egl_display = display;
			}
		}
		else
		{
			fprintf(stderr, "GetDisplay %d failed with error: %x\n", m_data->m_renderDevice, eglGetError());
		}

	}

	if (!eglInitialize(m_data->egl_display, NULL, NULL))
	{
		fprintf(stderr, "eglInitialize() failed with error: %x\n", eglGetError());
		exit(EXIT_FAILURE);
	}

	egl_version = gladLoaderLoadEGL(m_data->egl_display);
	if (!egl_version)
	{
		fprintf(stderr, "Unable to reload EGL.\n");
		exit(EXIT_FAILURE);
	}
	printf("Loaded EGL %d.%d after reload.\n", GLAD_VERSION_MAJOR(egl_version),
	       GLAD_VERSION_MINOR(egl_version));

	m_data->success = eglBindAPI(EGL_OPENGL_API);
	if (!m_data->success)
	{
		// TODO: Properly handle this error (requires change to default window
		// API to change return on all window types to bool).
		fprintf(stderr, "Failed to bind OpenGL API.\n");
		exit(EXIT_FAILURE);
	}

	m_data->success =
	    eglChooseConfig(m_data->egl_display, egl_config_attribs,
	                    &m_data->egl_config, 1, &m_data->num_configs);
	if (!m_data->success)
	{
		// TODO: Properly handle this error (requires change to default window
		// API to change return on all window types to bool).
		fprintf(stderr, "Failed to choose config (eglError: %d)\n", eglGetError());
		exit(EXIT_FAILURE);
	}
	if (m_data->num_configs != 1)
	{
		fprintf(stderr, "Didn't get exactly one config, but %d\n", m_data->num_configs);
		exit(EXIT_FAILURE);
	}

	m_data->egl_surface = eglCreatePbufferSurface(
	    m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);
	if (m_data->egl_surface == EGL_NO_SURFACE)
	{
		fprintf(stderr, "Unable to create EGL surface (eglError: %d)\n", eglGetError());
		exit(EXIT_FAILURE);
	}

        // Request a context with an OpenGL version of 3.3
        // The best resource I found for this was:
        // https://chromium.googlesource.com/angle/angle/+/refs/heads/master/src/libANGLE/renderer/gl/egl/DisplayEGL.cpp
        EGLint requestedMajor = 3;
        EGLint requestedMinor = 3;
        std::vector<EGLint> contextAttribList = {EGL_CONTEXT_MAJOR_VERSION, requestedMajor,
                                      EGL_CONTEXT_MINOR_VERSION, requestedMinor, EGL_NONE};

	m_data->egl_context = eglCreateContext(
            m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, contextAttribList.data());
	if (!m_data->egl_context)
	{
		fprintf(stderr, "Unable to create EGL context (eglError: %d)\n", eglGetError());
		exit(EXIT_FAILURE);
	}

	m_data->success =
	    eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface,
	                   m_data->egl_context);
	if (!m_data->success)
	{
		fprintf(stderr, "Failed to make context current (eglError: %d)\n", eglGetError());
		exit(EXIT_FAILURE);
	}

	if (!gladLoadGL((GLADloadfunc)eglGetProcAddress))
	{
		fprintf(stderr, "failed to load GL with glad.\n");
		exit(EXIT_FAILURE);
	}

	// print this in plugin:
	const GLubyte* ven = glGetString(GL_VENDOR);
	printf("GL_VENDOR=%s\n", ven);
	EGLAttrib cudaIndex;

    eglQueryDeviceAttribEXT(egl_devices[m_data->m_renderDevice], EGL_CUDA_DEVICE_NV, &cudaIndex);
    if (eglGetError() == EGL_SUCCESS) {
    	printf("CUDA_DEVICE=%d\n" , (int) cudaIndex);
    }
	const GLubyte* ren = glGetString(GL_RENDERER);
	printf("GL_RENDERER=%s\n", ren);
	const GLubyte* ver = glGetString(GL_VERSION);
	printf("GL_VERSION=%s\n", ver);
	const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
	printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);


	glViewport(0,0,m_data->m_windowWidth, m_data->m_windowHeight);
	//int i = pthread_getconcurrency();
	//printf("pthread_getconcurrency()=%d\n", i);
}

void EGLOpenGLWindow::closeWindow()
{
	eglMakeCurrent(m_data->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
	               EGL_NO_CONTEXT);
	eglDestroySurface(m_data->egl_display, m_data->egl_surface);
	eglDestroyContext(m_data->egl_display, m_data->egl_context);
	printf("Destroy EGL OpenGL window.\n");
}

void EGLOpenGLWindow::runMainLoop() {}

float EGLOpenGLWindow::getTimeInSeconds() { return 0.; }

bool EGLOpenGLWindow::requestedExit() const { return false; }

void EGLOpenGLWindow::setRequestExit() {}

void EGLOpenGLWindow::startRendering()
{
	// printf("EGL window start rendering.\n");
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
}

void EGLOpenGLWindow::endRendering()
{
	// printf("EGL window end rendering.\n");
	eglSwapBuffers(m_data->egl_display, m_data->egl_surface);
}


void EGLOpenGLWindow::setWindowTitle(const char* title) {}

float EGLOpenGLWindow::getRetinaScale() const { return 1.f; }

void EGLOpenGLWindow::setAllowRetina(bool allow) {}

int EGLOpenGLWindow::getWidth() const { return m_data->m_windowWidth; }

int EGLOpenGLWindow::getHeight() const { return m_data->m_windowHeight; }

int EGLOpenGLWindow::fileOpenDialog(char* fileName, int maxFileNameLength)
{
	return 0;
}

int  main(){
	std::cout << "Starting EGL query" << std::endl;

	EGLOpenGLWindow window = EGLOpenGLWindow();
	b3gWindowConstructionInfo ci = b3gWindowConstructionInfo();

	window.createWindow(ci);

    std::cout << "Completeing EGL query" << std::endl;
    return 0;
}
