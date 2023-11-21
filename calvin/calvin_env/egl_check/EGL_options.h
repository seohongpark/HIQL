struct b3gWindowConstructionInfo
{
	int m_width;
	int m_height;
	bool m_fullscreen;
	int m_colorBitsPerPixel;
	void* m_windowHandle;
	const char* m_title;
	int m_openglVersion;
	int m_renderDevice;

	b3gWindowConstructionInfo(int width = 1024, int height = 768)
	    : m_width(width),
	      m_height(height),
	      m_fullscreen(false),
	      m_colorBitsPerPixel(32),
	      m_windowHandle(0),
	      m_title("title"),
	      m_openglVersion(3),
	      m_renderDevice(-1)
	{
	}
};

class EGLOpenGLWindow
{
	struct EGLInternalData2* m_data;
	bool m_OpenGLInitialized;
	bool m_requestedExit;

public:
	EGLOpenGLWindow();
	virtual ~EGLOpenGLWindow();

	virtual void createDefaultWindow(int width, int height, const char* title)
	{
		b3gWindowConstructionInfo ci(width, height);
		ci.m_title = title;
		createWindow(ci);
	}

	virtual void createWindow(const b3gWindowConstructionInfo& ci);

	virtual void closeWindow();

	virtual void runMainLoop();
	virtual float getTimeInSeconds();

	virtual bool requestedExit() const;
	virtual void setRequestExit();

	virtual void startRendering();

	virtual void endRendering();


	virtual void setWindowTitle(const char* title);

	virtual float getRetinaScale() const;
	virtual void setAllowRetina(bool allow);

	virtual int getWidth() const;
	virtual int getHeight() const;

	virtual int fileOpenDialog(char* fileName, int maxFileNameLength);
};
