const supported_oses = ["Windows", "Linux"];

function hideAllBut(os, div=null) {
    let jq = div === null ? document: div;
    supported_oses.forEach(function(item, idx) {
        var selector = ".os-code-block ." + item.toLowerCase()
        $(jq).find(selector).hide();
        $(jq).find(".os-code-block li").removeClass("active");
    });

    $(jq).find(".os-code-block ." + os.toLowerCase()).show();
    $(jq).find('.os-code-block li:contains("' + os + '")').addClass("active");
}

function getOS() {
    var userAgent = window.navigator.userAgent,
        platform = window.navigator.platform,
        macosPlatforms = ['Macintosh', 'MacIntel', 'MacPPC', 'Mac68K'],
        windowsPlatforms = ['Win32', 'Win64', 'Windows', 'WinCE'],
        iosPlatforms = ['iPhone', 'iPad', 'iPod'],
        os = null;

    if (macosPlatforms.indexOf(platform) !== -1) {
        os = 'Mac OS';
    } else if (iosPlatforms.indexOf(platform) !== -1) {
        os = 'iOS';
    } else if (windowsPlatforms.indexOf(platform) !== -1) {
        os = 'Windows';
    } else if (/Android/.test(userAgent)) {
        os = 'Android';
    } else if (/Linux/.test(platform)) {
        os = 'Linux';
    }

    return os;
}


(function ($) {
    $(document).ready(function () {
        var os = getOS();

        if (os == null) {
             os = 'Windows';
        }

        $(".os-code-block div.choices li").on("click", function(){
            var os = $(this).text();
            $(this).class
            hideAllBut(os, $(this).closest("div.os-code-block").parent());
        });

        if (supported_oses.includes(os)) {
            hideAllBut(os);
        }
    });
})(jQuery);
