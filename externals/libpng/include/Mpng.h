/*
 * COPYRIGHT NOTICE, DISCLAIMER, and LICENSE:
 *
 * If you modify lib you may insert additional notices immediately following
 * this sentence.
 *
 * This code is released under the lib license.
 *
 * lib versions 1.0.7, July 1, 2000 through 1.6.29, March 16, 2017 are
 * Copyright (c) 2000-2002, 2004, 2006-2017 Glenn Randers-Pehrson, are
 * derived from lib-1.0.6, and are distributed according to the same
 * disclaimer and license as lib-1.0.6 with the following individuals
 * added to the list of Contributing Authors:
 *
 *    Simon-Pierre Cadieux
 *    Eric S. Raymond
 *    Mans Rullgard
 *    Cosmin Truta
 *    Gilles Vollant
 *    James Yu
 *    Mandar Sahastrabuddhe
 *    Google Inc.
 *    Vadim Barkov
 *
 * and with the following additions to the disclaimer:
 *
 *    There is no warranty against interference with your enjoyment of the
 *    library or against infringement.  There is no warranty that our
 *    efforts or the library will fulfill any of your particular purposes
 *    or needs.  This library is provided with all faults, and the entire
 *    risk of satisfactory quality, performance, accuracy, and effort is with
 *    the user.
 *
 * Some files in the "contrib" directory and some configure-generated
 * files that are distributed with lib have other copyright owners and
 * are released under other open source licenses.
 *
 * lib versions 0.97, January 1998, through 1.0.6, March 20, 2000, are
 * Copyright (c) 1998-2000 Glenn Randers-Pehrson, are derived from
 * lib-0.96, and are distributed according to the same disclaimer and
 * license as lib-0.96, with the following individuals added to the list
 * of Contributing Authors:
 *
 *    Tom Lane
 *    Glenn Randers-Pehrson
 *    Willem van Schaik
 *
 * lib versions 0.89, June 1996, through 0.96, May 1997, are
 * Copyright (c) 1996-1997 Andreas Dilger, are derived from lib-0.88,
 * and are distributed according to the same disclaimer and license as
 * lib-0.88, with the following individuals added to the list of
 * Contributing Authors:
 *
 *    John Bowler
 *    Kevin Bracey
 *    Sam Bushell
 *    Magnus Holmgren
 *    Greg Roelofs
 *    Tom Tanner
 *
 * Some files in the "scripts" directory have other copyright owners
 * but are released under this license.
 *
 * lib versions 0.5, May 1995, through 0.88, January 1996, are
 * Copyright (c) 1995-1996 Guy Eric Schalnat, Group 42, Inc.
 *
 * For the purposes of this copyright and license, "Contributing Authors"
 * is defined as the following set of individuals:
 *
 *    Andreas Dilger
 *    Dave Martindale
 *    Guy Eric Schalnat
 *    Paul Schmidt
 *    Tim Wegner
 *
 * The PNG Reference Library is supplied "AS IS".  The Contributing Authors
 * and Group 42, Inc. disclaim all warranties, expressed or implied,
 * including, without limitation, the warranties of merchantability and of
 * fitness for any purpose.  The Contributing Authors and Group 42, Inc.
 * assume no liability for direct, indirect, incidental, special, exemplary,
 * or consequential damages, which may result from the use of the PNG
 * Reference Library, even if advised of the possibility of such damage.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * source code, or portions hereof, for any purpose, without fee, subject
 * to the following restrictions:
 *
 *   1. The origin of this source code must not be misrepresented.
 *
 *   2. Altered versions must be plainly marked as such and must not
 *      be misrepresented as being the original source.
 *
 *   3. This Copyright notice may not be removed or altered from any
 *      source or altered source distribution.
 *
 * The Contributing Authors and Group 42, Inc. specifically permit, without
 * fee, and encourage the use of this source code as a component to
 * supporting the PNG file format in commercial products.  If you use this
 * source code in a product, acknowledgment is not required but would be
 * appreciated.
 *
 * END OF COPYRIGHT NOTICE, DISCLAIMER, and LICENSE.
 *
 * TRADEMARK:
 *
 * The name "lib" has not been registered by the Copyright owner
 * as a trademark in any jurisdiction.  However, because lib has
 * been distributed and maintained world-wide, continually since 1995,
 * the Copyright owner claims "common-law trademark protection" in any
 * jurisdiction where common-law trademark is recognized.
 *
 * OSI CERTIFICATION:
 *
 * Libpng is OSI Certified Open Source Software.  OSI Certified Open Source is
 * a certification mark of the Open Source Initiative. OSI has not addressed
 * the additional disclaimers inserted at version 1.0.7.
 *
 * EXPORT CONTROL:
 *
 * The Copyright owner believes that the Export Control Classification
 * Number (ECCN) for lib is EAR99, which means not subject to export
 * controls or International Traffic in Arms Regulations (ITAR) because
 * it is open source, publicly available software, that does not contain
 * any encryption software.  See the EAR, paragraphs 734.3(b)(3) and
 * 734.7(b).
 */

/*
 * A "png_get_copyright" function is available, for convenient use in "about"
 * boxes and the like:
 *
 *    printf("%s", png_get_copyright(NULL));
 *
 * Also, the PNG logo (in PNG format, of course) is supplied in the
 * files "pngbar.png" and "pngbar.jpg (88x31) and "pngnow.png" (98x31).
 */

/*
 * The contributing authors would like to thank all those who helped
 * with testing, bug fixes, and patience.  This wouldn't have been
 * possible without all of you.
 *
 * Thanks to Frank J. T. Wojcik for helping with the documentation.
 */

#ifndef _LPNG_H_
#define _LPNG_H_

#include "png.h"
#include "pngconf.h"
#include "pngdebug.h"
#include "pnginfo.h"
#include "pnglibconf.h"
#include "pngpriv.h"
#include "pngstruct.h"

#endif //LIBFL_LPNG_H
