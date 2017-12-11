// Parses the URL to extract the Get params Key-Value pairs.
function getUrlVars() {
  var vars = {};
  var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi,    
    function(m,key,value) {
      vars[key] = value;
    }
  );
  return vars;
}

// Uses GET Params to take an appropriate action to modify the country list.
function processURLAction() {
  urlVars = getUrlVars();
  if (typeof urlVars.country === "string" && urlVars.country.length > 0) {
    modifyCountry(urlVars.country, urlVars.action); 
    console.log(urlVars.country);
  }
}

// Modifies the localStorage for the the countries list.
function modifyCountry(countryName, action) {
  // by passing an object you can define default values e.g.: []
  chrome.storage.local.get({countries: []}, function (result) {
      // the input argument is ALWAYS an object containing the queried keys
      // so we select the key we need
      var countries = result.countries;
      if (action === "add") {
        console.log("Add");
        if(!countries.find(function(country){return country == countryName})) {
          countries.push(countryName);
        }
      } else {
        console.log("Remove");
        countries = countries.filter(function(x) {return x !== countryName})
      }
      // set the new array value to the same key
      chrome.storage.local.set({countries: countries}, function () {});
  });
}

// Loads the country names from the local storage and modifies the HTML page accordingly.
function loadCountryListHtml() {
  chrome.storage.local.get({countries: []}, function (result) {
      var html = "";
      var countries = result.countries;
      for(idx in countries) {
        var num = parseInt(idx) + 1;
        var country = countries[idx];
        html = html + "<tr><td>" + String(num) + "</td><td>" + country.replace('+', ' ') + "</td></tr>";
      }
      var myelement = document.getElementById("tableContent");
      myelement.innerHTML= html;

  }); 
}

// Main function taking actions before the page is loaded.
document.addEventListener('DOMContentLoaded', function() {
  console.log("Currently at " + document.URL);
  processURLAction();
  loadCountryListHtml();
}, false);

// chrome.storage.local.remove('countries')