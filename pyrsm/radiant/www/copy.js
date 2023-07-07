function copyToClipboard(copyText) {
    navigator.clipboard.writeText(copyText).then(function () {
        // alert('Copying to clipboard was successful!');
    }, function (err) {
        console.error('Could not copy text: ', err);
        alert('Could not copy text: ', err);
    });
}
