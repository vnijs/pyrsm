function copyToClipboard(copyText) {
    navigator.clipboard.writeText(copyText).then(function () {
        // alert('Copying to clipboard was successful!');
        // console.error('Copied text: ', copyText);
        // reset check mark to copy icon after 2 seconds
        setTimeout(function () {
            Shiny.setInputValue('copy_reset', Math.random());
        }, 2000);
    }, function (err) {
        console.error('Could not copy text: ', err);
        alert('Could not copy text: ', err);
    });
}
