var ipCountryCache = null;
var urlCountryCache = null;

var blockedCountries = null;

chrome.storage.local.get({countries: []}, function (result) {
  // the input argument is ALWAYS an object containing the queried keys
  // so we select the key we need
  blockedCountries = new Set();
  var arr = result.countries;
  for(var i=0; i < arr.length; i++) {
   blockedCountries.add(arr[i].replace('+', ' '));
  }
});

var expirationSlidingSeconds = 600;

function getIPtoCountry(domain_name){
  var xhr1 = new XMLHttpRequest();

  xmlurl = (
    "http://pro.viewdns.info/traceroute/?domain=" + 
    domain_name + 
    "&apikey=90a8769bbf7601f0da938c8f8c3dc172c72dd278&output=xml"
  );
  xhr1.open("GET", xmlurl, false);
  xhr1.send(null);
  xml1 = (xhr1.responseXML);
  ips = xml1.getElementsByTagName("ip");

  var countrySet = new Set();

  for (i = 0; i< ips.length; i++) {
    curr_ip = ips[i].childNodes[0].nodeValue;
    
    // Check cache. Only perform if missing in cache.
    var country = ipCountryCache.getItem(curr_ip);
    if (country == null) {
      url = "http://ip-api.com/xml/" + curr_ip;

      var xhr2 = new XMLHttpRequest();
      xhr2.open("GET", url, false);
      xhr2.send(null);
      xml2 = (xhr2.responseXML);

      var countries =  xml2.getElementsByTagName("country");
      var country;
      if(countries.length > 0){
        country = countries[0].childNodes[0].nodeValue;
        console.log("IPCountry Cache Miss. Saving: " + curr_ip + " - " + country);
        ipCountryCache.setItem(curr_ip, country, {expirationSliding: expirationSlidingSeconds});
      }
    } else {
      console.log("IPCountry Cache Hit: " + curr_ip + " - " + country);
    }
    if (country != null) {
      countrySet.add(country);
    }
  }
  return countrySet;
}

function processURL(domain_name) {

  // 1. Check UrlCountry cache for countries
  countriesList = urlCountryCache.getItem(domain_name);
  
  // 2. Get new country map
  if(countriesList == null) {
    var countrySet = getIPtoCountry(domain_name);
    for (var val of countrySet.values()) {
      //console.log(domain_name + "-->" + val);
    }
    countriesList = Array.from(countrySet);

    // 2.1 Save the country cache.
    console.log("DomainCountry Cache Miss. Saving: " + domain_name + " - " + countriesList);
    urlCountryCache.setItem(domain_name, countriesList, {expirationSliding: expirationSlidingSeconds});
  } else {
    console.log("DomainCountry Cache Hit: " + domain_name + " - " + countriesList);
  }

  // 3. Block countries.
  //console.log(blockedCountries);
  //console.log(countriesList);
  for (var country in countriesList) {
    if (blockedCountries.has(countriesList[country]) === true) {
      return countriesList[country];
    }
  }
  return null;
}

function initializeCache() {
  if (ipCountryCache == null) {
    ipCountryCache = new Cache(-1, false, new Cache.LocalStorageCacheStorage('IpCountry'));
  }
  if (urlCountryCache == null) {
    urlCountryCache = new Cache(-1, false, new Cache.LocalStorageCacheStorage('URLCountry'));
  }
}

chrome.webRequest.onBeforeRequest.addListener(
  function(details) {
    initializeCache();
    var hostname = details.url.split('/', 3)[2];

    var domain_name = new URL(details.url).hostname;

    blockingCountry = processURL(domain_name);
    // console.log(blockingCountry);
    if (blockingCountry != null) {
      var options = {
        type: "basic",
        title: "URL BLOCKED",
        message: "Visits blocked country: " + blockingCountry,
        iconUrl: "img/icon.png"
      }
      chrome.notifications.create("sd", options, function() {});
      return {
        cancel: true
      };
    }
  },
  {
      urls:["*://*/*"],
      types: ["main_frame"]
  },
  ["blocking"]
);