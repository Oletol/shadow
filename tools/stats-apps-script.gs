/**
 * Research statistics of the shadowing trainer in a Google Sheet.
 *
 * Set-up (once, about five minutes):
 *  1. Create a new Google Sheet. Menu Extensions - Apps Script.
 *  2. Delete the sample code, paste this file, press Save.
 *  3. Deploy - New deployment - type "Web app".
 *       Execute as: Me.   Who has access: Anyone.
 *     Press Deploy and allow the access Google asks for.
 *  4. Copy the "Web app URL" (it ends with /exec) into data/settings.json:
 *       "statsEndpoint": "https://script.google.com/macros/s/.../exec"
 *  5. Publish the site. Every attempt appears as a new row of the sheet.
 *
 * To stop collecting, set "statsEndpoint": null.
 * The sheet holds no names and no audio: a random code of the browser, scores and
 * the word-by-word detail (column "detail", JSON).
 */
var COLUMNS = ['time', 'anon', 'source', 'version', 'audio', 'phrase', 'text', 'step', 'attempt',
  'overall', 'melody', 'sounds', 'rhythm', 'stress', 'words', 'contour', 'syllables', 'tone',
  'errors', 'listens', 'duration', 'strictness', 'ipa', 'detail', 'trend'];

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    var d = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) sheet.appendRow(COLUMNS);
    sheet.appendRow(COLUMNS.map(function (k) {
      var v = d[k];
      if (v == null) return '';
      if (typeof v === 'object') v = JSON.stringify(v);
      v = String(v).slice(0, 5000);
      // a value must never be read by the sheet as a formula
      return /^[=+\-@]/.test(v) && isNaN(Number(v)) ? "'" + v : v;
    }));
  } finally {
    lock.releaseLock();
  }
  return ContentService.createTextOutput('ok');
}
