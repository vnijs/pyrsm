
// based on https://stackoverflow.com/questions/61690502/shiny-setinputvalue-only-works-on-the-2nd-try
function get_img_src() {
    var img_src = $("#screenshot_preview img").attr("src");
    Shiny.setInputValue("img_src", img_src);
}

function generate_screenshot() {
    var clonedHeight = document.querySelector('body').scrollHeight;
    html2canvas($("body")[0], {
        y: 55, // Set the starting point to 100 pixels from the top
        // width: document.body.scrollWidth,
        height: document.body.scrollHeight, // set this on the cloned document
        onclone: (clonedDocument) => {
            Array.from(clonedDocument.querySelectorAll("textarea")).forEach((textArea) => {
                if (textArea && textArea.value.length > 30) {
                    const labelFor = textArea.getAttribute("id")
                    const label = clonedDocument.querySelector(`label[for="${labelFor}"]`)
                    const div = clonedDocument.createElement("div")
                    div.innerText = textArea.value
                    div.style.border = "1px solid #d3d3d3"
                    div.style.padding = "10px 10px 10px 10px"
                    div.style.width = "100%"
                    div.style.borderRadius = "5px"
                    div.style.boxSizing = "border-box";
                    div.style.margin = "0";
                    div.style.backgroundColor = "white"
                    textArea.style.display = "none"
                    textArea.parentElement.append(label, div);
                }
            })

            Array.from(clonedDocument.querySelectorAll('select[multiple]')).forEach((msel) => {
                const multiSelect = document.querySelector("#" + msel.getAttribute("id"));
                if (multiSelect && multiSelect.selectedOptions.length > 1) {
                    const clonedMultiSelect = clonedDocument.querySelector("#" + msel.getAttribute("id"));
                    const list = clonedDocument.createElement('ul')
                    Array.from(multiSelect.selectedOptions).forEach((option) => {
                        const item = clonedDocument.createElement('li')
                        item.innerHTML = option.value
                        item.style = "list-style: none; padding-left: 0.5em"
                        item.style.width = "100%"
                        list.appendChild(item)
                    })
                    list.style.border = "1px solid #d3d3d3"
                    list.style.padding = "5px 5px 5px 5px"
                    list.style.width = "100%"
                    list.style.backgroundColor = "white"
                    list.style.borderRadius = "5px"
                    clonedMultiSelect.style.display = "none"
                    clonedMultiSelect.parentElement.appendChild(list)
                }
            });
            console.log(clonedDocument.querySelector("body").scrollHeight);
            clonedHeight = clonedDocument.querySelector("body").scrollHeight + "px";
            console.log("clonedHeight: " + clonedHeight);
        },
        ignoreElements: function (el) {
            return el.classList.contains("navbar-inverse") || el.classList.contains("dropdown-menu");
        }
    }).then(canvas => {
        var img = document.createElement("img");
        img.src = canvas.toDataURL("png");
        img.width = parseInt(canvas.style.width);
        img.height = parseInt(canvas.style.height); // changing value has no impact
        // has no impact even when "height:" above is not set
        // img.height = parseInt(clonedHeight);
        $("#screenshot_preview").empty();
        $("#screenshot_preview").append(img);
    });
}