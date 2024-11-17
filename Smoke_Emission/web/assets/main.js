async function handleTerminal() {

    await new Promise(resolve => setTimeout(resolve, 1500));

    let applies = [
        "(f1) Applying CLAHE",
        "(f2) Applying Anisotropic Diffusion",
        "(f3) Applying Top-hat and Bottom-hat transformations",
        "(f4) Applying Laplacian of Gaussian (LoG)",
        "finished applying filters!"
    ];
    let misc = [
        "(p1) Extracting LBP (Local Binary Patterns)",
        "(p2) Using Gabor filters for texture analysis",
        "(p3) Drawing red overlay",
        "(p4) Extracting B64 of image",
        "finished processing!"
    ]

    let brele = document.createElement('br');

    let terminalElement = document.getElementById('terminal');
    let lineElement1 = document.createElement('span');
    lineElement1.className = 'line aline italic';
    lineElement1.innerHTML = 'Initializing...';
    lineElement1.style.paddingBottom = '0.4rem'
    terminalElement.appendChild(lineElement1);
    terminalElement.scrollTop = terminalElement.scrollHeight - terminalElement.clientHeight

    for (let i = 0; i < applies.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 1200));
        lineElement1.innerHTML = applies[i];
        terminalElement.appendChild(lineElement1);
    }
    lineElement1.className = 'line sline';
    lineElement1.style.paddingBottom = '0'

    let lineElement2 = document.createElement('span');
    lineElement2.className = 'line aline italic';
    lineElement2.style.paddingBottom = '0.4rem'
    lineElement2.innerHTML = 'Processing...';
    terminalElement.appendChild(lineElement2);
    terminalElement.scrollTop = terminalElement.scrollHeight - terminalElement.clientHeight

    for (let i = 0; i < misc.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        lineElement2.innerHTML = misc[i];
        terminalElement.appendChild(lineElement2);
    }
    lineElement2.className = 'line sline';
    lineElement2.style.paddingBottom = '0'

}

async function getPathToFile() {
    document.getElementsByClassName("containerMain")[0].style.cursor = "wait";
    let terminalElement = document.getElementById('terminal');
    //let n = await eel.handleImage()();
    //handleTerminal();

    let [n] = await Promise.all([eel.handleImage()(), handleTerminal()]);

    if (n != "None") {
        let lineElement = document.createElement('div');
        lineElement.className = 'line bline sline';
        lineElement.innerHTML = "Success! Smoke detected in Image!";
        lineElement.style.paddingBottom = '0.4rem'
        terminalElement.appendChild(lineElement);
        let canvasElement = document.getElementById('canvasMain');
        canvasElement.src = "data:image/jpeg;base64," + n;
        canvasElement.style.display = "block";
    
    } else {
        let lineElement = document.createElement('div');
        lineElement.className = 'line bline eline';
        lineElement.innerHTML = "No smoke detected.";
        lineElement.style.paddingBottom = '0.4rem'
        terminalElement.appendChild(lineElement);
    }
    terminalElement.scrollTop = terminalElement.scrollHeight - terminalElement.clientHeight;
    document.getElementsByClassName("containerMain")[0].style.cursor = "default";
}