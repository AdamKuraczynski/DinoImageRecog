#include <windows.h>
#include <string.h>

const char g_szClassName[] = "myWindowClass";
char szClassName[] = "PRZYKLAD";

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_LBUTTONDOWN:
        {
            //char szFileName[MAX_PATH];
            //HINSTANCE hInstance = GetModuleHandle(NULL);

            //GetModuleFileName(hInstance, szFileName, MAX_PATH);
            //MessageBox(hwnd, szFileName, "This program is:", MB_OK | MB_ICONINFORMATION);
        }
        break;
        case WM_CLOSE:
            DestroyWindow(hwnd);
        break;
        case WM_DESTROY:
            PostQuitMessage(0);
        break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

struct
{
TCHAR * szClass;
int iStyle ;
TCHAR * szText ;
} button[] =
{
"BUTTON" , BS_PUSHBUTTON , "PUSHBUTTON",
"BUTTON" , BS_AUTOCHECKBOX , "CHECKBOX",
"BUTTON" , BS_RADIOBUTTON , "RADIOBUTTON",
"BUTTON" , BS_GROUPBOX , "GROUPBOX",
"EDIT" , WS_BORDER , "TEXTBOX",
"STATIC" , WS_BORDER , "STATIC",
} ;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
    WNDCLASSEX wc;
    HWND hwnd;
    MSG Msg;

    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = 0;
    wc.lpfnWndProc   = WndProc;
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = hInstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = g_szClassName;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);
    wc.cbSize = sizeof(WNDCLASSEX);

    if(!RegisterClassEx(&wc))
    {
        MessageBox(NULL, "Window Registration Failed!", "Error!",
            MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }
    
    


        
    hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        g_szClassName,
		"PRZYKLAD",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		HWND_DESKTOP,
		NULL,
		hInstance,
		NULL
	);Z

    if(hwnd == NULL)
    {
        MessageBox(NULL, "Window Creation Failed!", "Error!",
            MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    while(GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }
    return Msg.wParam;
}

LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
static HWND hwndButton[NUM];
static int cxChar, cyChar;
static RECT r;
HDC hdc;
int i;
PAINTSTRUCT ps;
TCHAR szFormat[] = TEXT ("%-16s Akcja: %04X, ID:%04X, hWnd:%08X");
TCHAR szBuffer[80];
switch (message)
{
case WM_CREATE :
cxChar = LOWORD (GetDialogBaseUnits ()) ;
cyChar = HIWORD (GetDialogBaseUnits ()) ;
for (i = 0 ; i < NUM ; i++)
hwndButton[i] = CreateWindow ( button[i].szClass,
button[i].szText,
WS_CHILD | WS_VISIBLE | button[i].iStyle,
cxChar, cyChar * (1 + 2 * i),
20 * cxChar, 7 * cyChar / 4,
hwnd, (HMENU) i,
((LPCREATESTRUCT) lParam)->hInstance, NULL) ;
break;
case WM_DESTROY:
PostQuitMessage(0);
break;
case WM_SIZE:
xSize = LOWORD(lParam);
ySize = HIWORD(lParam);
r.left = 24 * cxChar ;
r.top = 2 * cyChar ;
r.right = LOWORD (lParam) ;
r.bottom = HIWORD (lParam) ;
break;
case WM_COMMAND:
hdc = GetDC (hwnd);
ScrollWindow (hwnd, 0, -cyChar, &r, &r) ;
SelectObject (hdc, GetStockObject (SYSTEM_FIXED_FONT)) ;
SetBkMode (hdc, TRANSPARENT) ;
TextOut (hdc, 24 * cxChar, cyChar * (r.bottom / cyChar - 1),
szBuffer,
wsprintf (szBuffer, szFormat,
"WM_COMMAND",
HIWORD (wParam), LOWORD (wParam), lParam ));
ReleaseDC( hwnd, hdc );
return DefWindowProc(hwnd, message, wParam, lParam);
case WM_PAINT:
hdc = BeginPaint (hwnd, &ps);
EndPaint( hwnd, &ps );
break;
default:
return DefWindowProc(hwnd, message, wParam, lParam);
}
return 0;
}


